#!/usr/bin/env python
"""Mechanism study on simulated bearing signals with known ground truth (MSSP: "why/how").

Sub-commands
------------
  analytic   No training. Fixed band-pass FIR h: compares h*x (linear), ReLU(h*x) and (h*x)^2
             (square law) -> spectral prominence of the fault frequency at the layer output.
  train      Trains PhyQ-TransNet (quad=True, MED off == final model "phyq_nomed") and its
             linear twin (quad=False == "noquad") on simulated data, evaluates under white and
             impulsive noise.  Identical data, schedule and hyper-parameters for both.
  analyze    Fault-frequency prominence of the first-layer feature maps of the trained models
             (and of randomly initialised twins as a control).

All outputs go to <project>/results/sim_mechanism/ (nothing outside the project folder).

Examples
--------
  python scripts/sim_mechanism.py analytic
  python scripts/sim_mechanism.py train --seeds 0 1 2 --epochs 30 --n-train 1000
  python scripts/sim_mechanism.py analyze
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phyq.sim.bearing_sim import (CLASSES, SimConfig, add_impulsive, add_white,  # noqa: E402
                                  make_dataset, simulate_one)

OUT = ROOT / "results" / "sim_mechanism"
FS = 12_000.0
L = 2048
FAULTS = ("outer", "inner", "ball")


# ============================================================ prominence metric
def fault_prominence(sig: np.ndarray, fs: float, f_fault: float, n_harm: int = 3,
                     band=(15.0, 600.0)) -> np.ndarray:
    """Power at the first n_harm harmonics of f_fault (+/-1 bin) divided by the median power of
    the other bins in ``band`` -- computed on the (mean-removed) signal itself, i.e. WITHOUT any
    demodulation by the analyst.  Returns dB, shape sig.shape[:-1]."""
    x = sig - sig.mean(axis=-1, keepdims=True)
    win = np.hanning(x.shape[-1])
    P = np.abs(np.fft.rfft(x * win, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    df = f[1] - f[0]
    inband = (f >= band[0]) & (f <= band[1])
    harm = np.zeros_like(inband)
    for k in range(1, n_harm + 1):
        harm |= np.abs(f - k * f_fault) <= 1.01 * df
    harm &= inband
    bg = inband & ~harm
    num = P[..., harm].mean(axis=-1)
    den = np.median(P[..., bg], axis=-1) + 1e-30
    return 10 * np.log10(num / den + 1e-30)


def analysis_set(n: int, seed: int, snr_db=None):
    """Windows at a FIXED speed (1797 rpm) so that every fault frequency is known exactly."""
    rng = np.random.default_rng(seed)
    cfg = SimConfig()
    out = {}
    for c in FAULTS:
        xs, ff = [], None
        for _ in range(n):
            x, info = simulate_one(c, rng, cfg, rpm=1797.0, return_info=True)
            xs.append(x)
            ff = info["f_fault"]
        X = np.stack(xs)[:, None, :]
        if snr_db is not None:
            X = add_white(X, snr_db, rng)
        out[c] = (X, ff)
    return out


# ============================================================ analytic (no training)
def cmd_analytic(args):
    from scipy.signal import firwin, hilbert, lfilter
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for taps in (15, 63):
        h = firwin(taps, [2_400, 4_200], pass_zero=False, fs=FS)
        for snr in (None, 0, -6):
            sets = analysis_set(200, seed=11, snr_db=snr)
            for c, (X, ff) in sets.items():
                x = X[:, 0]
                y = lfilter(h, 1.0, x, axis=-1)
                env = np.abs(hilbert(y, axis=-1))
                r = {
                    "taps": taps, "snr": "clean" if snr is None else snr, "class": c, "f_fault": round(ff, 2),
                    "raw_input": float(np.median(fault_prominence(x, FS, ff))),
                    "linear_h*x": float(np.median(fault_prominence(y, FS, ff))),
                    "relu(h*x)": float(np.median(fault_prominence(np.maximum(y, 0), FS, ff))),
                    "square_(h*x)^2": float(np.median(fault_prominence(y ** 2, FS, ff))),
                    "hilbert_envelope": float(np.median(fault_prominence(env, FS, ff))),
                }
                rows.append(r)
    # ---- Gaussian vs impulsive noise at the SAME power: output distortion and fault prominence
    rows_imp = []
    rng = np.random.default_rng(5)
    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=FS)
    sets = analysis_set(200, seed=12)
    detectors = {"linear": lambda y: y, "relu": lambda y: np.maximum(y, 0), "square": lambda y: y ** 2,
                 "abs": np.abs}
    for snr in (6, 0, -6):
        for kind in ("white", "impulsive"):
            for c, (X, ff) in sets.items():
                x = X[:, 0]
                p_sig = (x ** 2).mean(-1, keepdims=True)
                if kind == "white":
                    n = rng.standard_normal(x.shape)
                else:                                    # same model as phyq/data/noise.py ('impulsive')
                    n = 0.3 * rng.standard_normal(x.shape) + (rng.random(x.shape) < 0.01) * rng.standard_normal(x.shape) * 10
                n = n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p_sig / 10 ** (snr / 10))
                y0 = lfilter(h, 1.0, x, axis=-1); y1 = lfilter(h, 1.0, x + n, axis=-1)
                for d, fn in detectors.items():
                    o0, o1 = fn(y0), fn(y1)
                    o0c, o1c = o0 - o0.mean(-1, keepdims=True), o1 - o1.mean(-1, keepdims=True)
                    dist = 10 * np.log10(((o1c - o0c) ** 2).mean(-1) / (o0c ** 2).mean(-1))
                    rows_imp.append({"snr": snr, "noise": kind, "class": c, "detector": d,
                                     "distortion_db": float(np.median(dist)),
                                     "prominence_db": float(np.median(fault_prominence(o1, FS, ff)))})
    (OUT / "analytic_impulsive.json").write_text(json.dumps(rows_imp, indent=1))
    print("\nnoise | snr | detector | output distortion dB (mean over classes) | fault prominence dB")
    import collections
    agg = collections.defaultdict(list)
    for r in rows_imp:
        agg[(r["noise"], r["snr"], r["detector"])].append((r["distortion_db"], r["prominence_db"]))
    for k, v in agg.items():
        v = np.array(v)
        print(" | ".join(map(str, k)), f"| {v[:, 0].mean():6.1f} | {v[:, 1].mean():6.1f}")
    (OUT / "analytic.json").write_text(json.dumps(rows, indent=1))
    hdr = ["taps", "snr", "class", "raw_input", "linear_h*x", "relu(h*x)", "square_(h*x)^2", "hilbert_envelope"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[k]) if not isinstance(r[k], float) else f"{r[k]:6.1f}" for k in hdr))


# ============================================================ training
def _torch():
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


SIM_MODELS = {   # name -> (registry name, cfg)
    "quad": ("phyq_paper", {"quad": True, "stats": True, "prefilter": {"enabled": False}}),     # phyq_nomed
    "noquad": ("phyq_paper", {"quad": False, "stats": True, "prefilter": {"enabled": False}}),
    "classbd": ("classbd", {"physics_loss": False}),
    "classbd_phys": ("classbd", {"physics_loss": True}),
    "wdcnn": ("wdcnn", {}),
    "qcnn": ("qcnn", {}),
}


def build(name, seed: int):
    torch = _torch()
    from phyq.models import build_model
    if isinstance(name, bool):                       # backwards compatibility
        name = "quad" if name else "noquad"
    reg, cfg = SIM_MODELS[name]
    torch.manual_seed(seed)
    return build_model(dict(cfg, name=reg), 1, len(CLASSES), L)


def evaluate(model, X, y, bs=64):
    torch = _torch()
    model.eval()
    dev = next(model.parameters()).device
    pred = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pred.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    acc = float((pred == y).mean())
    f1s = []
    for c in range(len(CLASSES)):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"acc": acc, "macro_f1": float(np.mean(f1s))}


def test_suite(n_per_class: int, seed: int = 999):
    """Fixed test sets (same for every model)."""
    rng = np.random.default_rng(seed)
    Xc, y = make_dataset(n_per_class, seed=seed)
    suite = {"clean": Xc}
    for snr in (0, -6, -10):
        suite[f"white{snr}"] = add_white(Xc, snr, np.random.default_rng(seed + 100 + snr))
    suite["impulsive"] = add_impulsive(Xc, np.random.default_rng(seed + 7))
    suite["impulsive+white0"] = add_impulsive(add_white(Xc, 0, np.random.default_rng(seed + 8)),
                                              np.random.default_rng(seed + 9))
    return suite, y


def cmd_train(args):
    torch = _torch()
    import torch.nn.functional as F
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
                          else "cpu")
    print(f"device: {device}")
    Xte_suite, yte = test_suite(args.n_test)
    Xva, yva = make_dataset(args.n_val, seed=500)
    Xva_n = add_white(Xva, 0, np.random.default_rng(501))
    for seed in args.seeds:
        Xtr, ytr = make_dataset(args.n_train, seed=1000 + seed)
        for name in args.models:
            tag = OUT / f"{name}_seed{seed}"
            if (tag / "metrics.json").exists() and not args.force:
                print(f"skip {tag.name} (exists)"); continue
            tag.mkdir(parents=True, exist_ok=True)
            model = build(name, seed).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            steps = args.epochs * math.ceil(len(Xtr) / args.batch)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
            rng = np.random.default_rng(seed)
            best, best_state, hist = -1, None, []
            t0 = time.time()
            for ep in range(args.epochs):
                model.train()
                perm = rng.permutation(len(Xtr))
                for i in range(0, len(perm), args.batch):
                    idx = perm[i:i + args.batch]
                    if len(idx) < 4:
                        continue
                    xb = Xtr[idx].copy()
                    # same on-the-fly augmentation as the benchmark: white noise, p=0.8, SNR U(-10,10)
                    for j in range(len(xb)):
                        if rng.random() < 0.8:
                            xb[j:j + 1] = add_white(xb[j:j + 1], rng.uniform(-10, 10), rng)
                        xb[j] = np.roll(xb[j], rng.integers(0, L), axis=-1)
                    out = model(torch.from_numpy(xb).to(device))
                    loss = F.cross_entropy(out, torch.from_numpy(ytr[idx]).to(device), label_smoothing=0.05)
                    if getattr(model, "aux_loss", None) is not None:
                        loss = loss + model.aux_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step()
                v = 0.5 * (evaluate(model, Xva, yva)["macro_f1"] + evaluate(model, Xva_n, yva)["macro_f1"])
                hist.append(v)
                if v > best:
                    best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                print(f"[{name} s{seed}] ep {ep + 1}/{args.epochs} val {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            model.load_state_dict(best_state)
            res = {k: evaluate(model, X, yte) for k, X in Xte_suite.items()}
            res["val_best"] = best
            res["val_hist"] = hist
            res["config"] = vars(args) | {"seed": seed, "model": name}
            torch.save(best_state, tag / "model.pt")
            (tag / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
            print(name, seed, {k: round(v["acc"] * 100, 2) for k, v in res.items() if isinstance(v, dict) and "acc" in v})


# ============================================================ analysis of learned layers
def first_layer_maps(model, X):
    """Stem conv output (pre-BN = 'pre') and after BN+activation ('post'); for the quadratic layer
    also the two components: (Wr*x)(Wg*x) and Wb*x^2."""
    torch = _torch()
    from phyq.models.phyq_paper import QuadraticConv1d
    model.eval()
    xt = torch.from_numpy(X)
    with torch.no_grad():
        xs = (xt - xt.mean(-1, keepdim=True)) / (xt.std(-1, keepdim=True) + 1e-5)
        conv, bn, act = model.stem[0], model.stem[1], model.stem[2]
        pre = conv(xs)
        post = act(bn(pre))
        out = {"pre": pre.numpy(), "post": post.numpy()}
        if isinstance(conv, QuadraticConv1d):
            out["product_rg"] = (conv.conv_r(xs) * conv.conv_g(xs)).numpy()
            out["power_b"] = conv.conv_b(xs * xs).numpy()
    return out


def cmd_analyze(args):
    torch = _torch()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for snr in (None, 0):
        sets = analysis_set(args.n, seed=21, snr_db=snr)
        for c, (X, ff) in sets.items():
            base = {"snr": "clean" if snr is None else snr, "class": c}
            rows.append(base | {"model": "raw_input", "seed": -1, "layer": "input",
                                "mean_db": float(np.median(fault_prominence(X[:, 0], FS, ff))), "top10_db": np.nan})
            for name in ("quad", "noquad"):
                for seed in args.seeds:
                    for trained in (True, False):
                        model = build(name, seed)
                        p = OUT / f"{name}_seed{seed}" / "model.pt"
                        if trained:
                            if not p.exists():
                                continue
                            model.load_state_dict(torch.load(p, map_location="cpu"))
                        maps = first_layer_maps(model, X)
                        for layer, M in maps.items():
                            pr = fault_prominence(M, FS / 2, ff)          # stem stride 2 -> fs/2
                            ch = np.median(pr, axis=0)                     # per-channel median over windows
                            rows.append(base | {"model": name + ("" if trained else "_init"), "seed": seed,
                                                "layer": layer, "mean_db": float(ch.mean()),
                                                "top10_db": float(np.sort(ch)[-max(1, len(ch) // 10):].mean())})
    (OUT / "prominence.json").write_text(json.dumps(rows, indent=1))
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["snr"], r["class"], r["model"], r["layer"])].append((r["mean_db"], r["top10_db"]))
    print("snr | class | model | layer | mean dB | top-10% channels dB")
    for k, v in sorted(agg.items(), key=lambda kv: tuple(map(str, kv[0]))):
        v = np.array(v, dtype=float)
        print(" | ".join(map(str, k)), f"| {np.nanmean(v[:, 0]):6.1f} | {np.nanmean(v[:, 1]):6.1f}")


def cmd_summary(args):
    import collections
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(OUT.glob("*_seed*/metrics.json")):
        m = json.loads(p.read_text())
        name = p.parent.name.rsplit("_seed", 1)[0]
        for k, v in m.items():
            if isinstance(v, dict) and "acc" in v:
                acc[name][k].append(100 * v["acc"])
    conds = ["clean", "white0", "white-6", "white-10", "impulsive", "impulsive+white0"]
    lines = ["| model | n seeds | " + " | ".join(conds) + " |", "|---" * (len(conds) + 2) + "|"]
    for name, d in acc.items():
        n = len(d["clean"])
        lines.append(f"| {name} | {n} | " + " | ".join(
            f"{np.mean(d[c]):.1f} ± {np.std(d[c]):.1f}" if d[c] else "-" for c in conds) + " |")
    txt = "\n".join(lines)
    (OUT / "accuracy_summary.md").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analytic")
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=int, nargs="+", default=[0])
    t.add_argument("--models", nargs="+", default=["quad", "noquad"])
    t.add_argument("--n-train", type=int, default=400, help="windows per class")
    t.add_argument("--n-val", type=int, default=100)
    t.add_argument("--n-test", type=int, default=200)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--force", action="store_true")
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    sub.add_parser("summary")
    a = sub.add_parser("analyze")
    a.add_argument("--seeds", type=int, nargs="+", default=[0])
    a.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    {"analytic": cmd_analytic, "train": cmd_train, "analyze": cmd_analyze, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
