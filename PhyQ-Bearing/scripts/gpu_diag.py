"""GPU diagnosis for PhyQ-TransNet (v9.2). Every case runs in its own fresh process with
CUDA_LAUNCH_BLOCKING=1, so a crash in one case cannot affect the others and the failing kernel is named.

    python scripts\\gpu_diag.py            # ~3-6 min, prints a summary table at the end
    python scripts\\gpu_diag.py --quick    # only the two main cases
"""
import argparse
import json
import os
import subprocess
import sys
import time

import _bootstrap  # noqa: F401

CASES = [  # (model, window, micro-batch, amp)
    ("phyq_paper", 2048, 64, False),
    ("phyq_paper", 2048, 16, False),
    ("phyq_paper", 8192, 16, False),
    ("phyq_paper", 8192, 32, False),
    ("wdcnn", 2048, 64, True),
    ("wdcnn", 8192, 32, True),
]


def child(model_name, L, B, amp, safe):
    import numpy as np
    import torch
    from phyq.models import build_model
    from phyq.runner import apply_cuda_safe_mode
    from phyq.suite import build_run_config
    dev = torch.device("cuda")
    from phyq.utils.misc import seed_everything
    cfg = build_run_config("pu", model_name)
    seed_everything(0, bool(cfg.get("deterministic", True)))     # same cuDNN mode as the real runs
    apply_cuda_safe_mode(safe)
    m = build_model(cfg["model"], 1, 3, L)
    if hasattr(m, "fit_prefilter") and getattr(m, "prefilter", None) is not None:
        m.fit_prefilter(np.random.randn(64, 1, L).astype("float32"), device=dev)   # exercises cuFFT
    m.to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for it in range(6):
        if it == 1:
            torch.cuda.synchronize()
            t0 = time.time()
        x = torch.randn(B, 1, L, device=dev)
        y = torch.randint(0, 3, (B,), device=dev)
        with torch.autocast("cuda", enabled=amp):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    step = (time.time() - t0) / 5
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        m(torch.randn(64, 1, L, device=dev))
    torch.cuda.synchronize()
    print("RESULT " + json.dumps({"ok": True, "loss": float(loss), "s_per_step": step,
                                  "peak_GB": torch.cuda.max_memory_allocated() / 2**30}))


def run_case(model_name, L, B, amp, safe):
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--child", model_name, str(L), str(B), str(int(amp)), str(safe)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = [l for l in (p.stderr or "").strip().splitlines() if l.strip()]
    tail = " | ".join(err[-3:])[:600]
    where = [l.strip() for l in err if l.strip().startswith("File ") and "site-packages" not in l][-1:]
    return {"ok": False, "error": tail, "where": where[0] if where else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.child:
        mn, L, B, amp, safe = a.child
        child(mn, int(L), int(B), bool(int(amp)), int(safe))
        return
    import torch
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free {free / 2**30:.2f} / {total / 2**30:.2f} GB")
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout)
    except Exception as e:  # noqa: BLE001
        print("nvidia-smi not available:", e)
    cases = CASES[:3] if a.quick else CASES
    rows = []
    for mn, L, B, amp in cases:
        for safe in (0, 1, 2):
            t = time.time()
            r = run_case(mn, L, B, amp, safe)
            tag = f"{mn:10s} L={L:<5d} B={B:<3d} {'amp ' if amp else 'fp32'} safe={safe}"
            if r["ok"]:
                print(f"OK    {tag}  {r['s_per_step']:.3f} s/step  peak {r['peak_GB']:.2f} GB  ({time.time() - t:.0f}s)")
            else:
                print(f"FAIL  {tag}  {r['error']}\n        at: {r['where']}")
            rows.append(dict(model=mn, L=L, B=B, amp=amp, safe=safe, **r))
            if r["ok"] or "out of memory" in r.get("error", "").lower():
                break                       # OOM is not fixed by the safe modes
            time.sleep(3)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "gpu_diag.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("\nsaved results/gpu_diag.json  -> please send this whole output")


if __name__ == "__main__":
    main()
