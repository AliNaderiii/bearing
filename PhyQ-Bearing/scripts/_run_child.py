"""Internal: run ONE experiment in a fresh process (used by run_suite.py on the GPU).

A CUDA fault ('illegal memory access', raw 'CUDA error: out of memory') poisons the CUDA context of the
whole process; in a child process it only affects this run and the suite can retry it safely.
Exit codes: 0 ok, 2 data error, 3 CUDA out of memory, 4 CUDA fault, 1 other error.
"""
import pickle
import sys
import traceback

import _bootstrap  # noqa: F401


def main(job_path: str, status_path: str) -> int:
    import os
    os.environ["PHYQ_ISOLATED"] = "1"
    from phyq.runner import DataError, is_cuda_fault, is_cuda_oom, run_experiment
    from phyq.utils.misc import save_json
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    try:
        run_experiment(job["cfg"], job["out_dir"], job["seed"], force=job["force"])
        save_json({"kind": "ok"}, status_path)
        return 0
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        if isinstance(e, DataError):
            kind, code = "data", 2
        elif is_cuda_oom(e):
            kind, code = "oom", 3
        elif is_cuda_fault(e):
            kind, code = "cuda_fault", 4
        else:
            kind, code = "error", 1
        save_json({"kind": kind, "error": repr(e)[:2000],
                   "dataset_level": bool(getattr(e, "dataset_level", False))}, status_path)
        return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
