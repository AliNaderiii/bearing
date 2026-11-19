"""End-to-end inference cost (reviewer request): latency for batch sizes 1/8/64 on GPU and CPU
(1 thread = embedded proxy, all threads), peak GPU memory, and a per-module MAC breakdown.

The forward pass of every model already contains ALL processing after windowing: instance
normalisation, the statistical features (PhyQ-TransNet) and the classifier. No test-time augmentation
or ensembling is used anywhere in the MSSP manuscript, so one forward pass = one decision.

    python scripts/profile_e2e.py --out results/_paper_assets/profile_e2e.json
"""
import argparse
import copy
import json
import os
import time

import torch

import _bootstrap  # noqa: F401
from phyq.models import build_model
from phyq.runner import profile
from phyq.suite import CFG_DIR
from phyq.utils.config import load_config

MODELS = {   # tag -> (config, overrides)
    "phyq_nomed": ("phyq_paper", {"prefilter": {"enabled": False}}),
    "phyq_nomed_noquad": ("phyq_paper", {"prefilter": {"enabled": False}, "quad": False}),
    "classbd": ("classbd", {}),
    "wdcnn": ("wdcnn", {}),
    "qcnn": ("qcnn", {}),
    "drsn": ("drsn", {}),
    "cnn_transformer": ("cnn_transformer", {}),
}


def _cfg(tag):
    name, ov = MODELS[tag]
    cfg = load_config(os.path.join(CFG_DIR, "models", f"{name}.yaml"))["model"]
    for k, v in ov.items():
        cfg[k] = dict(cfg.get(k, {}), **v) if isinstance(v, dict) else v
    return cfg


@torch.no_grad()
def _latency(net, x, reps):
    dev = x.device
    for _ in range(5):
        net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


@torch.no_grad()
def _module_macs(net, C, L):
    """MACs of every top-level child, measured with thop on the input it receives in a real forward."""
    from thop import profile as thop_profile
    net = copy.deepcopy(net).cpu().eval()
    inputs = {}
    def keep(name):
        def hook(mod, inp, out):      # must return None (a returned value would replace the output)
            inputs.setdefault(name, inp)
        return hook

    hooks = [m.register_forward_hook(keep(n)) for n, m in net.named_children()]
    net(torch.randn(1, C, L))
    for h in hooks:
        h.remove()
    out = {}
    for n, m in net.named_children():
        if n not in inputs or not any(p.numel() for p in m.parameters()) and n != "transformer":
            continue
        try:
            macs, _ = thop_profile(copy.deepcopy(m), inputs=tuple(i for i in inputs[n] if torch.is_tensor(i)),
                                   verbose=False)
            out[n] = float(macs)
        except Exception as e:  # pragma: no cover
            out[n] = f"n/a ({type(e).__name__})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 8, 64])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--out", default="results/_paper_assets/profile_e2e.json")
    args = ap.parse_args()
    C, L, K = args.channels, args.length, args.classes
    res = {"input": [C, L], "classes": K, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads_all": torch.get_num_threads(), "models": {}}
    n_all = torch.get_num_threads()
    for tag in args.models:
        net = build_model(_cfg(tag), C, K, L).eval()
        r = dict(profile(net, C, L))
        r["module_macs"] = _module_macs(net, C, L)
        if torch.cuda.is_available():
            g = net.cuda()
            for b in args.batches:
                x = torch.randn(b, C, L, device="cuda")
                torch.cuda.reset_peak_memory_stats()
                ms = _latency(g, x, args.reps)
                r[f"gpu_bs{b}_ms"] = ms
                r[f"gpu_bs{b}_ms_per_window"] = ms / b
                r[f"gpu_bs{b}_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            net = g.cpu()
            torch.cuda.empty_cache()
        if not args.no_cpu:
            for th in (1, n_all):
                torch.set_num_threads(th)
                r[f"cpu{th}t_bs1_ms"] = _latency(net, torch.randn(1, C, L), max(5, args.reps // 3))
            torch.set_num_threads(n_all)
        res["models"][tag] = r
        print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "module_macs"})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
