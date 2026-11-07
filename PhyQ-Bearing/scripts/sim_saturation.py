#!/usr/bin/env python
"""Detector-level check (no training) of WHY ClassBD survives impulsive noise on the simulator although
its front-end is also quadratic: its quadratic filter output passes BN -> ReLU -> Sigmoid, i.e. a
SATURATING (compressive) law.  Fixed band-pass FIR (15 or 63 taps) + detector, simulated faults at a
fixed speed, the simulator's impulsive interference (phyq.sim.bearing_sim.add_impulsive).

Output distortion = pooled power of (detector(noisy) - detector(clean)) / power of detector(clean), dB.
Prominence = fault_prominence (scripts/sim_mechanism.py) of detector(noisy), median over windows.

  python scripts/sim_saturation.py      -> results/sim_mechanism/saturation.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phyq.sim.bearing_sim import add_impulsive  # noqa: E402

_spec = importlib.util.spec_from_file_location("sim_mechanism", ROOT / "scripts" / "sim_mechanism.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def main():
    sets = sm.analysis_set(200, seed=21)
    rows = []
    for taps in (15, 63):
        h = firwin(taps, [2_400, 4_200], pass_zero=False, fs=sm.FS)
        num, den, prom = {}, {}, {}
        for c, (X, ff) in sets.items():
            Xi = add_impulsive(X, np.random.default_rng(7))
            y0 = lfilter(h, 1.0, X[:, 0], axis=-1)
            y1 = lfilter(h, 1.0, Xi[:, 0], axis=-1)
            q0 = y0 ** 2
            mu, sd = q0.mean(), q0.std()          # BN-like standardisation with CLEAN statistics
            dets = {
                "linear": lambda y: y,
                "relu": lambda y: np.maximum(y, 0),
                "square": lambda y: y ** 2,
                "square_bn_relu_sigmoid": lambda y: 1 / (1 + np.exp(-np.maximum((y ** 2 - mu) / sd, 0))),
            }
            for d, f in dets.items():
                o0, o1 = f(y0), f(y1)
                o0c, o1c = o0 - o0.mean(-1, keepdims=True), o1 - o1.mean(-1, keepdims=True)
                num[d] = num.get(d, 0.0) + float(((o1c - o0c) ** 2).sum())
                den[d] = den.get(d, 0.0) + float((o0c ** 2).sum())
                prom.setdefault(d, []).append(float(np.median(sm.fault_prominence(o1, sm.FS, ff))))
        for d in num:
            r = {"taps": taps, "detector": d, "distortion_db": round(10 * np.log10(num[d] / den[d]), 2),
                 "prominence_db": round(float(np.mean(prom[d])), 2)}
            rows.append(r)
            print(f"{taps:3d} taps | {d:24s} | distortion {r['distortion_db']:6.1f} dB | prominence {r['prominence_db']:5.1f} dB")
    out = ROOT / "results" / "sim_mechanism"
    out.mkdir(parents=True, exist_ok=True)
    (out / "saturation.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
