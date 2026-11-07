"""One experiment = (dataset, split variant, model, seed). Used by scripts/run.py and run_suite.py."""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from .data.pipeline import PreparedData, prepare
from .engine.evaluate import evaluate_grid
from .engine.trainer import NonFiniteLoss, train
from .models import build_model, count_params
from .models.layers import fit_med
from .utils.config import dump_config, interpolate
from .utils.misc import environment_info, get_logger, resolve_device, save_json, seed_everything, stable_hash

_DATA_CACHE: Dict[str, PreparedData] = {}


class DataError(RuntimeError):
    """Raised when a dataset cannot be found/read/split (affects every model on that data)."""


def get_data(cfg: Dict, logger=None) -> PreparedData:
    key = stable_hash(cfg["data"], 16)
    if key not in _DATA_CACHE:
        _DATA_CACHE.clear()  # keep at most one dataset in memory
        import gc
        gc.collect()
        try:
            _DATA_CACHE[key] = prepare(cfg, seed=0, logger=logger)
        except Exception as e:
            err = DataError(f"{type(e).__name__}: {e}")
            # files/folders missing -> every variant of this dataset will fail the same way
            err.dataset_level = isinstance(e, FileNotFoundError) or "No recordings" in str(e)
            raise err from e
    return _DATA_CACHE[key]


def profile(model, in_ch, L) -> Dict:
    out = {"params": count_params(model)}
    try:
        from thop import profile as thop_profile
        m = copy.deepcopy(model).cpu().eval()
        macs, _ = thop_profile(m, inputs=(torch.randn(1, in_ch, L),), verbose=False)
        out["macs"] = float(macs) + _attention_macs(copy.deepcopy(model).cpu().eval(), in_ch, L)
    except Exception as e:  # pragma: no cover
        out["macs"] = None
        out["macs_error"] = str(e)
    return out


@torch.no_grad()
def _attention_macs(model, in_ch, L) -> float:
    """thop ignores the functional parts of nn.MultiheadAttention (in/out projections and
    QK^T, AV products). Add them analytically: 4*T*d^2 + 2*T^2*d per attention layer."""
    tot = [0.0]

    def hook(mod, inp, out):
        q = inp[0]
        T, d = (q.shape[1], q.shape[2]) if mod.batch_first else (q.shape[0], q.shape[2])
        tot[0] += 4 * T * d * d + 2 * T * T * d

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    model(torch.randn(1, in_ch, L))
    for h in hs:
        h.remove()
    return tot[0]


@torch.no_grad()
def latency_ms(model, in_ch, L, device, reps=30) -> float:
    model.eval()
    x = torch.randn(1, in_ch, L, device=device)
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000



CLASSICAL = {"kurtogram_ses"}   # non-gradient baselines (phyq/classical)


def _build_and_train(cfg, data, device, seed, logger, K, L):
    if cfg["model"]["name"].lower() in CLASSICAL:
        from .classical.kurtogram import fit_classical
        return fit_classical(cfg, data, device, seed, logger, K)
    model = build_model(cfg["model"], data.n_channels, K, L)

    mcfg = cfg["model"].get("med", {}) or {}
    if mcfg.get("enabled", False) and hasattr(model, "set_med"):
        sample = data.train.materialize(max_items=int(mcfg.get("fit_windows", 64)), seed=seed)
        filt = np.stack([fit_med(sample[:, c], int(mcfg.get("length", 32)), int(mcfg.get("iters", 30)))
                         for c in range(sample.shape[1])])
        model.set_med(filt)

    pf = cfg["model"].get("prefilter", {}) or {}
    if hasattr(model, "fit_prefilter") and getattr(model, "prefilter", None) is not None:
        # Neural-MED of PhyQ-TransNet: fitted on TRAINING windows only, then frozen
        sample = data.train.materialize(max_items=int(pf.get("max_windows", 2000)), seed=seed)
        model.fit_prefilter(sample, seed=seed, device=device, logger=logger)
        del sample

    prof = profile(model, data.n_channels, L)
    logger.info(f"[run] {cfg['data']['name']} | {cfg.get('_variant', '')} | {cfg['model']['name']} | seed {seed} | "
                f"params {prof['params']:,} | device {device}")
    plan_micro_batch(model, cfg, device, data.n_channels, L, logger)
    tr = train(model, data, cfg, device, seed, logger, K)
    return model, prof, tr


def is_cuda_oom(e: BaseException) -> bool:
    """torch's allocator OOM *and* raw 'CUDA error: out of memory' raised by cuDNN/cuBLAS kernels."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def is_cuda_fault(e: BaseException) -> bool:
    msg = str(e)
    return isinstance(e, RuntimeError) and not is_cuda_oom(e) and any(
        k in msg for k in ("CUDA error", "CUDNN_STATUS", "CUBLAS_STATUS", "cuFFT error", "CUFFT_",
                           "illegal memory access", "device-side assert",
                           # cuDNN v9 benchmark ('FIND') mode on GTX 16xx at some conv shapes (v9.4)
                           "FIND was unable to find an engine", "cuDNN error", "cudnn"))


def apply_cuda_safe_mode(level: int, logger=None) -> None:
    """Work-arounds for driver/kernel faults on small consumer GPUs (numerically equivalent maths):
    1 = cuDNN heuristic algorithm choice instead of benchmark/'FIND' mode + plain 'math' attention
        kernel (no fused SDPA kernels) + no cuFFT plan cache
    2 = additionally cuDNN off (native PyTorch convolutions; slower).
    Must be called AFTER seed_everything (which sets cudnn.benchmark)."""
    if level <= 0:
        return
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        torch.backends.cuda.cufft_plan_cache.max_size = 0
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    if level >= 2:
        torch.backends.cudnn.enabled = False
    if logger:
        logger.warning(f"  [cuda-safe level {level}] cuDNN heuristics (no benchmark), math attention, no cuFFT plan cache"
                       + (", cuDNN disabled" if level >= 2 else ""))


def _probe_train_peak(model, C: int, L: int, b: int, amp: bool, device) -> int:
    """Peak GPU bytes of one forward+backward with micro-batch b (on a copy: BatchNorm stats untouched)."""
    import gc
    rng = torch.cuda.get_rng_state(device)           # the probe must not shift the training RNG stream
    m = copy.deepcopy(model).to(device).train()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    x = torch.randn(b, C, L, device=device)
    with torch.autocast(device_type="cuda", enabled=amp):
        out = m(x)
    out.float().logsumexp(1).mean().backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    del m, x, out
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_rng_state(rng, device)
    return int(peak)


def plan_micro_batch(model, cfg: Dict, device, C: int, L: int, logger=None) -> None:
    """v9.2: choose gradient accumulation BEFORE training from a measured memory footprint, instead of
    crashing with CUDA OOM mid-run (PhyQ-TransNet, 8192 samples, float32: ~75 MB/sample -> batch 32 needs
    ~2.4 GB of activations, too close to the limit of a 4 GB laptop GPU). Effective batch is unchanged."""
    tcfg = cfg["train"]
    if device.type != "cuda" or not tcfg.get("auto_micro_batch", True):
        return
    bs = int(tcfg.get("batch_size", 64))
    min_micro = min(bs, int(tcfg.get("min_micro_batch", 16)))
    amp = bool(tcfg.get("amp", True))
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    try:
        peak = _probe_train_peak(model, C, L, min_micro, amp, device)
    except Exception as e:
        if is_cuda_oom(e):
            raise RuntimeError(f"CUDA out of memory already with micro-batch {min_micro} "
                               f"(free {free / 2**30:.2f} of {total / 2**30:.2f} GB). Close other programs "
                               f"that use the GPU (browser, games...) or use a shorter data.window for ALL "
                               f"models (fairness).") from e
        raise
    per = peak / min_micro
    # reserve for cuDNN workspaces + allocator fragmentation. v9.4: 0.80 -> 0.55 after a mid-run OOM at
    # 1.5 of 3.0 GB free (GTX 1650); micro-batch 16x2 instead of 32x1 cost only ~4 % time there.
    budget = 0.55 * free - 256 * 2**20
    acc = max(1, int(tcfg.get("grad_accum", 1)))
    while bs // acc > min_micro and per * (bs // acc) > budget:
        acc *= 2
    if per * (bs // acc) > budget and logger:
        # v9.5: the probe itself just ran fine with this micro-batch, so it physically fits. The estimate
        # is pessimistic (it includes cuDNN benchmark workspaces: WDCNN/QCNN under AMP measured ~1.5 GB
        # for micro-batch 16 although their activations are far smaller) -> warn and continue; a real
        # OOM later is still caught by the retry logic.
        logger.warning(f"  [memory] estimate {per * (bs // acc) / 2**30:.2f} GB > budget {budget / 2**30:.2f} GB at "
                       f"the minimum micro-batch {bs // acc}; continuing (the probe with this micro-batch succeeded)")
    if logger:
        logger.info(f"  [memory] ~{per / 2**20:.0f} MB/sample (train), free {free / 2**30:.2f}/{total / 2**30:.2f} GB "
                    f"-> micro-batch {bs // acc} x accum {acc} (effective batch {bs})")
    tcfg["grad_accum"] = acc


def _seed(cfg: Dict, seed: int, logger=None) -> None:
    seed_everything(seed, bool(cfg.get("deterministic", True)))
    apply_cuda_safe_mode(int(cfg.get("cuda_safe", 0)), logger)


def run_experiment(cfg: Dict, out_dir: str, seed: int, force: bool = False) -> Optional[Dict]:
    cfg = interpolate(cfg)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path) and not force:
        return None
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("phyq", os.path.join(out_dir, "train.log"))
    dump_config(cfg, os.path.join(out_dir, "config.yaml"))
    t_run0 = time.time()
    device = resolve_device(cfg.get("device", "auto"))
    if int(cfg.get("cuda_safe", 0)) < int(cfg.get("_test_fault_below_safe", 0)):     # unit tests only
        raise RuntimeError("CUDA error: an illegal memory access was encountered (simulated)")
    data = get_data(cfg, logger)
    _seed(cfg, seed, logger)
    L = int(cfg["data"]["window"])
    K = len(data.class_names)
    for attempt in range(5):
        try:
            model, prof, tr = _build_and_train(cfg, data, device, seed, logger, K, L)
            break
        except NonFiniteLoss as e:
            if not e.amp:
                raise RuntimeError(f"{e}: training diverged in float32 (no AMP); result would be "
                                   f"meaningless -> run stopped") from e
            # float16 overflow (e.g. quadratic convolutions on long windows): restart in float32
            model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cfg["train"]["amp"] = False
            logger.warning(f"  {e} -> restarting this run from scratch WITHOUT AMP (float32)")
            _seed(cfg, seed, logger)
        except RuntimeError as e:
            if not is_cuda_oom(e):
                raise
            if not isinstance(e, torch.cuda.OutOfMemoryError) and os.environ.get("PHYQ_ISOLATED") == "1":
                # raw 'CUDA error: out of memory' (cuDNN/cuBLAS) may leave this process' CUDA context
                # unusable -> let run_suite retry in a FRESH process with a smaller micro-batch
                raise
            # GPU too small for this batch: same effective batch via gradient accumulation
            model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            acc = int(cfg["train"].get("grad_accum", 1)) * 2
            bs = int(cfg["train"].get("batch_size", 64))
            min_micro = int(cfg["train"].get("min_micro_batch", 16))
            if acc > 8 or bs // acc < min_micro:
                # v9: micro-batches < 16 break BatchNorm statistics. Verified on PU: PhyQ-TransNet
                # reaches ~97 % with micro-batch 16 but stays at chance (~45 %) with micro-batch 4.
                raise RuntimeError(
                    f"CUDA out of memory even with micro-batch {bs // (acc // 2)}; a smaller micro-batch "
                    f"(< train.min_micro_batch={min_micro}) would corrupt BatchNorm statistics and give "
                    f"meaningless results. Close other GPU programs, or use a shorter data.window for "
                    f"this dataset for ALL models (fairness).")
            cfg["train"]["grad_accum"] = acc
            logger.warning(f"  CUDA out of memory -> restarting this run with grad_accum={acc} "
                           f"(micro-batch {int(cfg['train'].get('batch_size', 64)) // acc}, same effective batch)")
            _seed(cfg, seed, logger)
    prof["latency_ms_bs1"] = latency_ms(model, data.n_channels, L, device)

    ecfg = cfg.get("eval", {})
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if cfg.get("_val_only", False):
        # architecture search: the test set is never touched
        vres = evaluate_grid(model, data.val, device, K, dict(ecfg, snrs=cfg.get("selection", {}).get("report_snrs",
                             [-6, 0]), snr_modes=["relative"], noise_types=["white"], noise_seed=777),
                             data.ref_power, amp, keep_probs=False)
        out = {"dataset": cfg["data"]["name"], "variant": cfg.get("_variant"), "model": cfg["model"]["name"],
               "model_tag": cfg.get("_model_tag"), "seed": seed, "val_only": True, "val": vres,
               "train": {k: v for k, v in tr.items() if k != "history"}, "profile": prof, "data": data.summary}
        save_json(tr["history"], os.path.join(out_dir, "history.json"))
        out["wall_time_s"] = time.time() - t_run0
        save_json(out, metrics_path)
        logger.info(f"[done/val-only] best val {tr['best_val']:.2f}@{tr['best_epoch']}")
        return out
    res = evaluate_grid(model, data.test, device, K, ecfg, data.ref_power, amp)
    probs = res.pop("_probs")
    out = {
        "dataset": cfg["data"]["name"], "variant": cfg.get("_variant"), "model": cfg["model"]["name"],
        "model_tag": cfg.get("_model_tag", cfg["model"]["name"]), "seed": seed, "test": res,
        "train": {k: v for k, v in tr.items() if k != "history"}, "profile": prof, "data": data.summary,
        "env": environment_info(), "cuda_safe": int(cfg.get("cuda_safe", 0)),
        "grad_accum": int(cfg["train"].get("grad_accum", 1)),
    }
    if int(ecfg.get("tta_shifts", 0)) > 1:
        r2 = evaluate_grid(model, data.test, device, K, ecfg, data.ref_power, amp,
                           tta_shifts=int(ecfg["tta_shifts"]), keep_probs=False)
        out["test_tta"] = r2
    if getattr(model, "last_alpha", None) is not None:
        out["fusion_alpha_last_batch_mean"] = model.last_alpha.mean(0).cpu().tolist()
        out["fusion_streams"] = getattr(model, "streams", None)
    save_json(tr["history"], os.path.join(out_dir, "history.json"))
    np.savez_compressed(os.path.join(out_dir, "test_probs.npz"), y=data.test.labels,
                        rec_idx=data.test.index[:, 0], **{k.replace("|", "__"): v for k, v in probs.items()})
    if cfg.get("save_checkpoint", False):
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    out["wall_time_s"] = time.time() - t_run0
    save_json(out, metrics_path)
    _run_extra_tests(cfg, data, model, device, K, ecfg, amp, out_dir, out, logger)
    c = res["clean"]
    logger.info(f"[done] clean acc {c['acc']:.2f} F1 {c['macro_f1']:.2f} | best val {tr['best_val']:.2f}"
                f"@{tr['best_epoch']} | {tr['train_time_s']:.0f}s")
    return out


def _window_ids(ds, recs):
    return [(recs[int(ri)].rid, int(st)) for ri, st in ds.index]


def _run_extra_tests(cfg, data, model, device, K, ecfg, amp, out_dir, out, logger) -> None:
    """Evaluate the trained model on the test sets of variants with an IDENTICAL train/val split.

    Results are written as normal runs (``<exp>/<dataset>/<variant>/<model>/<seed>/metrics.json``,
    flagged ``trained_in``) so tables and resume logic treat them like any other run. If the
    train/val windows or the normalisation differ in any way, the extra test is refused.
    """
    extra = cfg.get("_extra_tests") or {}
    if not extra:
        return
    ref_train, ref_val = _window_ids(data.train, data.recordings), _window_ids(data.val, data.recordings)
    seed_dir, tag_dir = os.path.basename(out_dir), os.path.basename(os.path.dirname(out_dir))
    ds_dir = os.path.dirname(os.path.dirname(os.path.dirname(out_dir)))
    for name, dcfg in extra.items():
        sub_dir = os.path.join(ds_dir, name, tag_dir, seed_dir)
        mpath = os.path.join(sub_dir, "metrics.json")
        if os.path.exists(mpath):
            continue
        sub_cfg = dict(cfg, data=copy.deepcopy(dcfg), _variant=name)
        sub_cfg.pop("_extra_tests", None)
        try:
            d2 = get_data(sub_cfg, logger)
        except DataError as e:
            logger.warning(f"  [extra-test] {name}: data error {e} -> skipped")
            continue
        same = (_window_ids(d2.train, d2.recordings) == ref_train and _window_ids(d2.val, d2.recordings) == ref_val
                and np.allclose(d2.train.scale, data.train.scale) and d2.class_names == data.class_names)
        if not same:
            logger.warning(f"  [extra-test] {name}: train/val split differs from {cfg.get('_variant')} "
                           f"-> NOT evaluated (run it as a normal variant)")
            continue
        r2 = evaluate_grid(model, d2.test, device, K, ecfg, data.ref_power, amp)
        probs2 = r2.pop("_probs", {})
        os.makedirs(sub_dir, exist_ok=True)
        np.savez_compressed(os.path.join(sub_dir, "test_probs.npz"), y=d2.test.labels,
                            rec_idx=d2.test.index[:, 0], **{k.replace("|", "__"): v for k, v in probs2.items()})
        dump_config(sub_cfg, os.path.join(sub_dir, "config.yaml"))
        o2 = dict(out, variant=name, test=r2, data=d2.summary, trained_in=cfg.get("_variant"))
        o2.pop("test_tta", None)
        save_json(o2, mpath)
        logger.info(f"  [extra-test] {name}: clean acc {r2['clean']['acc']:.2f} F1 {r2['clean']['macro_f1']:.2f} "
                    f"(same model as {cfg.get('_variant')})")

