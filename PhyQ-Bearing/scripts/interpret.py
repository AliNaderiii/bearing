#!/usr/bin/env python
"""Interpretability of the trained PhyQ-TransNet on REAL data (Fig. 10/11 of the MSSP manuscript).

Step 1 (GPU, once): train with checkpoints, same configuration and seed as main_tuned
    python scripts/run_suite.py configs/experiments/interp.yaml --datasets cwru
    python scripts/run_suite.py configs/experiments/interp.yaml --datasets sdust --models phyq_nomed
Step 2 (GPU or CPU, minutes): inspect the saved weights
    python scripts/interpret.py analyze                 # every results/interp/**/model.pt
    python scripts/interpret.py analyze --max-windows 3000
Step 3: figures (also works on another computer that only has the small analysis files)
    python scripts/pro_figures.py --only interp

For every trained model, ``analyze`` stores in results/interp/_analysis/<dataset>__<variant>__<tag>.npz:
  * the learned kernels of the first (stem) layer (W_r, W_g, W_b for the quadratic model, W for the twin);
  * class-averaged power spectra (0 - fs/4) of the raw input, of the stem output before batch normalisation,
    of its bilinear (product) and power terms, of the stem output after BN + activation, and the two
    classical references (squared Hilbert envelope and Teager-Kaiser energy of the raw input), for clean
    windows and for white (0 dB), impulsive and pink noise (-6 dB, absolute SNR, paired evaluation noise);
  * the fusion gate alpha (weight of the statistical stream) of every test window under each condition;
  * the RMS activation (log10) of every convolutional stage, the 1x1 fusion and z_d, per window and condition;
  * the deep, statistical and fused representations (z_d, z_s, z) of a fixed random subset of test windows
    for t-SNE, with labels and predictions.
Nothing outside results/interp/ is written. Data are read through the normal pipeline (cache inside the
project), test windows only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

# evaluation conditions: (name, noise kind, SNR mode, SNR dB)
CONDS = [("clean", None, None, None), ("white_0", "white", "absolute", 0.0),
         ("white_-6", "white", "absolute", -6.0), ("white_-10", "white", "absolute", -10.0),
         ("pink_-6", "pink", "absolute", -6.0), ("impulsive_-6", "impulsive", "absolute", -6.0)]
SPEC_CONDS = ("clean", "white_0", "impulsive_-6", "pink_-6")
EMB_CONDS = ("clean", "white_-10", "impulsive_-6")
SPEC_FMAX = 800.0          # Hz kept for the feature-map spectra (the raw-input spectrum is kept in full)
# CWRU drive-end bearing SKF 6205-2RS: fault orders (multiples of the shaft frequency) and speeds per load
CWRU_ORDERS = {"OR": 3.5848, "IR": 5.4152, "B": 2.3568}
CWRU_RPM = {"within_0HP": 1797.0, "within_1HP": 1772.0, "within_2HP": 1750.0, "within_3HP": 1730.0}


def _psd(y):
    """y: (B, C, T) torch -> (B, C, F) power spectrum of the de-meaned, Hann-windowed signal."""
    import torch
    y = y.float()
    y = y - y.mean(-1, keepdim=True)
    w = torch.hann_window(y.shape[-1], periodic=False, device=y.device)   # = np.hanning
    return torch.fft.rfft(y * w, dim=-1).abs().pow(2)


def fault_freq(class_name: str, rpm):
    """Characteristic fault frequency (Hz) of a CWRU class name such as 'OR007', None for Normal/unknown."""
    if rpm is None:
        return None
    for key in ("OR", "IR", "B"):
        if class_name.startswith(key):
            return CWRU_ORDERS[key] * rpm / 60.0
    return None


def prominence_from_psd(P, fs_rep: float, T: int, f_fault, n_harm: int = 3, band=(15.0, 600.0)):
    """Same metric as scripts/sim_mechanism.fault_prominence, from an existing Hann power spectrum.
    P: (B, C, F) numpy, f_fault: (B,) Hz (nan = no fault).  Returns (B, C) dB (nan where no fault)."""
    f = np.fft.rfftfreq(T, 1 / fs_rep)
    df = f[1] - f[0]
    inband = (f >= band[0]) & (f <= band[1])
    out = np.full(P.shape[:2], np.nan, np.float32)
    for i, ff in enumerate(f_fault):
        if not np.isfinite(ff):
            continue
        harm = np.zeros_like(inband)
        for k in range(1, n_harm + 1):
            harm |= np.abs(f - k * ff) <= 1.01 * df
        harm &= inband
        bg = inband & ~harm
        num = P[i][:, harm].mean(-1)
        den = np.median(P[i][:, bg], -1) + 1e-30
        out[i] = 10 * np.log10(num / den + 1e-30)
    return out


def analyze_run(run_dir: Path, out_dir: Path, max_windows: int, n_emb: int, device: str):
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from phyq.data.noise import add_noise, eval_generator
    from phyq.models import build_model
    from phyq.runner import get_data

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = get_data(cfg)
    K = len(data.class_names)
    L = int(cfg["data"]["window"])
    model = build_model(cfg["model"], data.n_channels, K, L)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(dev).eval()
    quad = hasattr(model.stem[0], "conv_r")
    ds_name, variant, tag = cfg["data"]["name"], cfg.get("_variant", ""), cfg.get("_model_tag", run_dir.parent.name)
    fs = float(data.fs)
    ecfg = cfg.get("eval", {})
    bs = int(ecfg.get("batch_size", 256))
    seed = int(ecfg.get("noise_seed", 12345))

    # ---- kernels of the stem layer
    out = {"class_names": np.array(data.class_names), "fs": fs, "quad": quad, "L": L,
           "n_channels": data.n_channels}
    st = model.stem[0]
    if quad:
        out["W_r"] = st.conv_r.weight.detach().cpu().numpy()
        out["W_g"] = st.conv_g.weight.detach().cpu().numpy()
        out["W_b"] = st.conv_b.weight.detach().cpu().numpy()
    else:
        out["W"] = st.weight.detach().cpu().numpy()

    # ---- hooks
    cap = {}
    hooks = [model.stem[0].register_forward_hook(lambda m, i, o: cap.__setitem__("stem_pre", o)),
             model.stem[2].register_forward_hook(lambda m, i, o: cap.__setitem__("stem_post", o))]
    if quad:
        hooks += [st.conv_r.register_forward_hook(lambda m, i, o: cap.__setitem__("r", o)),
                  st.conv_g.register_forward_hook(lambda m, i, o: cap.__setitem__("g", o)),
                  st.conv_b.register_forward_hook(lambda m, i, o: cap.__setitem__("power", o))]
    if getattr(model, "use_tf", False):
        hooks.append(model.transformer.register_forward_hook(lambda m, i, o: cap.__setitem__("z_d", o.mean(1))))
    if getattr(model, "use_stats", False):
        hooks.append(model.aux_net.register_forward_hook(lambda m, i, o: cap.__setitem__("z_s", o)))
    hooks.append(model.classifier.register_forward_hook(lambda m, i, o: cap.__setitem__("z", i[0])))
    # activation magnitude through depth (pre-normalisation outputs of every conv stage)
    depth = {"stem_conv": model.stem[0], "branch1_conv": model.branch_1[0], "branch2_conv": model.branch_2[0],
             "branch3_conv": model.branch_3[0], "fusion_conv": model.fusion_conv, "qres1": model.deep_stage[0],
             "qres2": model.deep_stage[2]}
    for dk, mod in depth.items():
        hooks.append(mod.register_forward_hook(lambda m, i, o, dk=dk: cap.__setitem__("rms_" + dk, o)))
    out["depth_layers"] = np.array(list(depth) + ["z_d"])

    test = data.test
    n = len(test)
    rng = np.random.default_rng(0)
    use = np.sort(rng.choice(n, size=min(n, max_windows), replace=False)) if max_windows and n > max_windows \
        else np.arange(n)
    emb_idx = np.sort(rng.choice(len(use), size=min(len(use), n_emb), replace=False))
    emb_mask = np.zeros(len(use), bool)
    emb_mask[emb_idx] = True
    sub = torch.utils.data.Subset(test, use.tolist())
    labels = np.asarray(test.labels)[use]
    out["labels"] = labels
    rpm = CWRU_RPM.get(variant) if ds_name == "cwru" else None
    ffc = np.array([fault_freq(c, rpm) or np.nan for c in data.class_names], float)
    out["fault_freq_per_class"] = ffc
    out["emb_labels"] = labels[emb_mask]

    for cname, kind, mode, snr in CONDS:
        dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=0)
        spec, prom = {}, {}
        alphas, preds, embs, rms = [], [], {"z_d": [], "z_s": [], "z": []}, []
        pos = 0
        with torch.no_grad():
            for b, (x, _) in enumerate(dl):
                if kind is not None:
                    g = eval_generator(seed, float(snr), kind, b)
                    x = add_noise(x, float(snr), kind, mode, data.ref_power, gen=g)
                x = x.to(dev)
                cap.clear()
                logits = model(x)
                yb = labels[pos:pos + len(x)]
                mb = emb_mask[pos:pos + len(x)]
                pos += len(x)
                preds.append(logits.argmax(-1).cpu().numpy())
                r = [cap["rms_" + dk].double().pow(2).flatten(1).mean(1).sqrt() for dk in depth]   # float64: no overflow
                r.append(cap["z_d"].double().pow(2).mean(1).sqrt() if "z_d" in cap else torch.full_like(r[0], np.nan))
                rms.append(torch.stack(r, 1).cpu().numpy())              # (B, n_layers) RMS per window
                if getattr(model, "last_alpha", None) is not None:
                    alphas.append(model.last_alpha.float().cpu().numpy().ravel())
                if cname in EMB_CONDS:
                    for k in embs:
                        if k in cap:
                            embs[k].append(cap[k].float().cpu().numpy()[mb])
                if cname in SPEC_CONDS:
                    reps = {"input": x, "stem_pre": cap["stem_pre"], "stem_post": cap["stem_post"]}
                    if quad:
                        reps["product"] = cap["r"] * cap["g"]
                        reps["power"] = cap["power"]
                    # analyst's reference: squared Hilbert envelope of the raw (noisy) input
                    Xf = torch.fft.fft(x.float(), dim=-1)
                    h = torch.zeros(x.shape[-1], device=x.device)
                    h[0] = 1
                    h[1:(x.shape[-1] + 1) // 2] = 2
                    if x.shape[-1] % 2 == 0:
                        h[x.shape[-1] // 2] = 1
                    reps["input_env"] = torch.fft.ifft(Xf * h, dim=-1).abs().pow(2)
                    # second classical square-law demodulator: Teager-Kaiser energy operator of the raw input
                    xf = x.float()
                    reps["input_tkeo"] = torch.nn.functional.pad(xf[..., 1:-1].pow(2) - xf[..., :-2] * xf[..., 2:], (1, 1))  # same length L
                    for rk, rv in reps.items():
                        P = _psd(rv).cpu().numpy()                       # (B, C, F)
                        if np.isfinite(ffc).any():
                            T = rv.shape[-1]
                            prom.setdefault(rk, []).append(prominence_from_psd(P, fs * T / L, T, ffc[yb]))
                        acc = spec.setdefault(rk, np.zeros((K,) + P.shape[1:], np.float64))
                        for k in np.unique(yb):
                            acc[k] += P[yb == k].sum(0)
        counts = np.bincount(labels, minlength=K).astype(float)
        nkeep = int(SPEC_FMAX * L / fs) + 1 if fs else None      # bin spacing is fs/L for every representation
        for rk, acc in spec.items():
            m = acc / np.maximum(counts, 1)[:, None, None]
            if rk != "input" and nkeep:
                m = m[..., :nkeep]                                # baseband only (keeps the files small)
            out[f"spec__{rk}__{cname}"] = m.astype(np.float32)
        for rk, v in prom.items():
            out[f"prom__{rk}__{cname}"] = np.concatenate(v)
        pr = np.concatenate(preds)
        out[f"pred__{cname}"] = pr
        out[f"log10rms__{cname}"] = np.log10(np.concatenate(rms) + 1e-300).astype(np.float32)
        out[f"acc__{cname}"] = float((pr == labels).mean() * 100)
        if alphas:
            out[f"alpha__{cname}"] = np.concatenate(alphas).astype(np.float32)
            out[f"nanfrac__{cname}"] = float(np.isnan(out[f"alpha__{cname}"]).mean())
        for k, v in embs.items():
            if v:
                out[f"emb_{k}__{cname}"] = np.concatenate(v).astype(np.float16)
        if cname in EMB_CONDS:
            out[f"emb_pred__{cname}"] = pr[emb_mask]
        print(f"  {ds_name}/{variant}/{tag}: {cname:13s} acc {out[f'acc__{cname}']:.1f}%"
              + (f"  mean alpha {np.nanmean(out[f'alpha__{cname}']):.3f}" if alphas else ""))
    for h in hooks:
        h.remove()
    if rpm:
        out["rpm"] = rpm
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{ds_name}__{variant}__{tag}.npz"
    np.savez_compressed(fn, **out)
    print("  wrote", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze"])
    ap.add_argument("--exp", default="interp")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--max-windows", type=int, default=4000, help="test windows used per model (random subset)")
    ap.add_argument("--n-emb", type=int, default=1500, help="windows kept for t-SNE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    base = Path(a.results) / a.exp
    out_dir = base / "_analysis"
    runs = sorted(Path(p).parent for p in glob.glob(str(base / "*" / "*" / "*" / "seed*" / "model.pt")))
    if not runs:
        sys.exit(f"no model.pt under {base} -> run first: python scripts/run_suite.py configs/experiments/{a.exp}.yaml")
    for r in runs:
        ds, var, tag = r.parts[-4], r.parts[-3], r.parts[-2]
        fn = out_dir / f"{ds}__{var}__{tag}.npz"
        if fn.exists() and not a.force:
            print("  exists, skipped:", fn.name)
            continue
        print(f"[analyze] {r}")
        analyze_run(r, out_dir, a.max_windows, a.n_emb, a.device)
    summary = {}
    for p in sorted(out_dir.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        summary[p.stem] = {k: float(z[k]) for k in z.files if k.startswith("acc__")}
        for k in z.files:
            if k.startswith("alpha__"):
                summary[p.stem][k + "_mean"] = float(np.nanmean(z[k]))
            if k.startswith("prom__"):
                ch = np.nanmedian(z[k], axis=0)                 # per-channel median over fault windows
                summary[p.stem][k + "_top10pct_db"] = float(np.sort(ch)[-max(1, len(ch) // 10):].mean())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("  wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
