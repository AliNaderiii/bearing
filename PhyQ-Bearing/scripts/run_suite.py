"""Run a whole experiment suite (resumable: finished runs are skipped).

    python scripts/run_suite.py configs/experiments/main_benchmark.yaml
    python scripts/run_suite.py configs/experiments/main_benchmark.yaml --datasets cwru pu --models phyq_transnet wdcnn
    python scripts/run_suite.py configs/experiments/smoke.yaml train.epochs=2      # extra overrides
    python scripts/run_suite.py configs/experiments/main_benchmark.yaml --dry      # list runs only
"""
import argparse
import os
import sys
import time
import traceback

import _bootstrap  # noqa: F401
from phyq.preflight import check_data_paths, report
from phyq.runner import DataError, run_experiment
from phyq.suite import expand
from phyq.utils.misc import save_json


def gpu_ok(allow_cpu: bool) -> bool:
    """Refuse to start hours of work on the CPU by accident (e.g. .venv not activated -> system Python)."""
    import torch
    if torch.cuda.is_available() or allow_cpu or os.environ.get("PHYQ_ALLOW_CPU") == "1":
        return True
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print("\n" + "!" * 70)
    print("NO CUDA GPU VISIBLE -> stopped (on the CPU this would take days).")
    print(f"  python used : {sys.executable}  (Python {sys.version.split()[0]}, torch {torch.__version__})")
    if not in_venv:
        print("  -> the project's .venv is NOT active. Run first:   .\\.venv\\Scripts\\Activate.ps1")
    else:
        print("  -> torch in this environment has no CUDA. Check: python scripts\\check_env.py")
    print("  (to really run on CPU add --allow-cpu)")
    print("!" * 70)
    return False


def run_isolated(s, force: bool, max_safe: int = 2) -> dict:
    """Run one experiment in a child process; retry CUDA faults with safer CUDA settings and raw CUDA
    out-of-memory errors with a smaller micro-batch. Returns the final status dict."""
    import copy
    import json
    import pickle
    import subprocess
    os.makedirs(s.out_dir, exist_ok=True)
    cfg = copy.deepcopy(s.cfg)
    job, status = os.path.join(s.out_dir, "_job.pkl"), os.path.join(s.out_dir, "_status.json")
    child = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_run_child.py")
    st = {"kind": "error", "error": "not started"}
    for attempt in range(6):
        with open(job, "wb") as f:
            pickle.dump({"cfg": cfg, "out_dir": s.out_dir, "seed": s.seed, "force": force or attempt > 0}, f)
        if os.path.exists(status):
            os.remove(status)
        rc = subprocess.call([sys.executable, child, job, status])
        try:
            with open(status, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            st = {"kind": "cuda_fault" if rc != 0 else "ok",
                  "error": f"child process ended with code {rc} without a status (driver crash?)"}
        if st["kind"] in ("ok", "data", "error"):
            break
        if st["kind"] == "cuda_fault":
            lvl = int(cfg.get("cuda_safe", 0)) + 1
            if lvl > max_safe:
                break
            cfg["cuda_safe"] = lvl
            print(f"!! CUDA fault -> retrying this run in a NEW process with cuda-safe level {lvl}")
        elif st["kind"] == "oom":
            t = cfg["train"]
            acc = int(t.get("grad_accum", 1)) * 2
            bs, mm = int(t.get("batch_size", 64)), int(t.get("min_micro_batch", 16))
            if bs // acc < mm:
                st["error"] += f" | micro-batch would drop below min_micro_batch={mm}: close other GPU programs"
                break
            t["grad_accum"] = acc
            print(f"!! CUDA out of memory -> retrying in a NEW process with micro-batch {bs // acc} x {acc}")
        time.sleep(5)        # let the driver release the old context
    for p in (job, status):
        if os.path.exists(p):
            os.remove(p)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="run even if no CUDA GPU is visible (very slow)")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run all experiments in this process (default on GPU: one fresh process per run)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    specs = list(expand(a.experiment, a.overrides, a.datasets, a.models, a.seeds))
    # group by dataset/variant so each dataset is prepared once
    specs.sort(key=lambda s: (s.dataset, s.variant, s.seed, s.model_tag))
    # skip runs whose resolved configuration is identical to another run (e.g. ablation 'no_envelope'
    # when the selected model has no envelope branch) -> no wasted GPU hours
    from phyq.utils.misc import stable_hash
    seen, uniq = {}, []
    for s in specs:
        key = stable_hash({k: v for k, v in s.cfg.items() if k != "_model_tag"}, 16) + f"|{s.seed}"
        if key in seen:
            print(f"  skip {s.dataset}/{s.variant}/{s.model_tag}/seed{s.seed}: identical to {seen[key]}")
            continue
        seen[key] = s.model_tag
        uniq.append(s)
    specs = uniq
    todo = [s for s in specs if a.force or not os.path.exists(os.path.join(s.out_dir, "metrics.json"))]
    import phyq
    print(f"PhyQ-Bearing code version {phyq.__version__}")
    print(f"{len(specs)} runs in suite, {len(todo)} to do")
    if a.dry:
        for s in todo:
            print(f"  {s.dataset:8s} {s.variant:14s} {s.model_tag:22s} seed{s.seed}")
        return
    if not todo:
        return 0
    if not gpu_ok(a.allow_cpu):
        return 1
    problems = check_data_paths({s.dataset for s in todo})
    if problems:
        print(report(problems))
        return 1
    import torch
    isolate = torch.cuda.is_available() and not a.no_isolate
    failures, bad_data = [], {}
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        key = (s.dataset, s.variant)
        if (s.dataset, "*") in bad_data:
            key = (s.dataset, "*")
        if key in bad_data:          # same data already failed -> do not repeat it for every model/seed
            failures.append({"run": s.out_dir, "error": "skipped: " + bad_data[key]})
            continue
        print(f"\n=== [{i}/{len(todo)}] {s.dataset} / {s.variant} / {s.model_tag} / seed{s.seed} "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)")
        if isolate:
            st = run_isolated(s, a.force)
            if st["kind"] == "ok":
                continue
            print(f"!! run FAILED ({st['kind']}): {st.get('error', '')[:400]}")
            failures.append({"run": s.out_dir, "kind": st["kind"], "error": st.get("error", "")})
            if st["kind"] == "data":
                key = (s.dataset, "*") if st.get("dataset_level") else key
                bad_data[key] = st.get("error", "")[:300]
                print(f"!! DATA problem for {key[0]}/{key[1]}: all its remaining runs are skipped")
            if a.stop_on_error:
                break
            continue
        try:
            run_experiment(s.cfg, s.out_dir, s.seed, force=a.force)
        except DataError as e:
            traceback.print_exc()
            if getattr(e, "dataset_level", False):
                key = (s.dataset, "*")
            bad_data[key] = str(e)[:300]
            print(f"!! DATA problem for {key[0]}/{key[1]}: all its remaining runs are skipped")
            failures.append({"run": s.out_dir, "error": repr(e)})
            if a.stop_on_error:
                raise
        except Exception as e:
            traceback.print_exc()
            failures.append({"run": s.out_dir, "error": repr(e)})
            if a.stop_on_error:
                raise
    if failures:
        os.makedirs("results", exist_ok=True)
        save_json(failures, os.path.join("results", f"failures_{int(time.time())}.json"))
        print(f"\n{len(failures)}/{len(todo)} runs FAILED (see results/failures_*.json)")
        for (ds, var), msg in bad_data.items():
            print(f"   data problem  {ds}/{var}: {msg}")
    print(f"Finished in {(time.time() - t0) / 60:.1f} min")
    # data problems or a majority of failures -> non-zero exit so run_all stops immediately
    return 1 if (bad_data or len(failures) > len(todo) // 2) else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
