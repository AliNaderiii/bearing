"""Params / MACs / latency of every model for a given input shape.

    python scripts/profile_models.py --channels 1 --length 2048 --classes 10
"""
import argparse
import os

import torch

import _bootstrap  # noqa: F401
from phyq.models import build_model
from phyq.runner import latency_ms, profile
from phyq.suite import CFG_DIR
from phyq.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=["phyq_paper", "wdcnn", "qcnn", "drsn",
                                                    "cnn_transformer", "stats_mlp"])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("| model | params | MACs (M) | latency bs=1 (ms, %s) |\n|---|---|---|---|" % dev.type)
    for m in a.models:
        cfg = load_config(os.path.join(CFG_DIR, "models", f"{m}.yaml"))["model"]
        net = build_model(cfg, a.channels, a.classes, a.length)
        p = profile(net, a.channels, a.length)
        lat = latency_ms(net.to(dev), a.channels, a.length, dev)
        print(f"| {m} | {p['params']:,} | {p['macs'] / 1e6:.1f} | {lat:.2f} |")


if __name__ == "__main__":
    main()
