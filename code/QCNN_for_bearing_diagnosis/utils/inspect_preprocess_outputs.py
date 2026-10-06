# -*- coding: utf-8 -*-
"""
inspect_preprocess_outputs.py
Inspect & summarise outputs produced by Preprocess_refactored_v3_production.py

Usage:
    python inspect_preprocess_outputs.py --out_dir "<path/to/processed_data_dir>" [--save_plots] [--verbose]

Example:
    python inspect_preprocess_outputs.py --out_dir "data/0HP/processed_data_robust" --save_plots
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import traceback
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------- helpers ----------------
def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load json {path}: {e}")
        return None

def find_npz(out_dir: str) -> Optional[str]:
    patterns = ["processed_data_*.npz", "processed_data*.npz"]
    matches = []
    for p in patterns:
        found = glob.glob(os.path.join(out_dir, p))
        if found:
            found.sort()
            matches.extend(found)
    if matches:
        return matches[-1]
    # fallback: any npz in folder
    any_npz = glob.glob(os.path.join(out_dir, "*.npz"))
    if any_npz:
        any_npz.sort()
        return any_npz[-1]
    return None

def safe_load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Try to load NPZ safely:
      - first try allow_pickle=False
      - if that fails, fall back to allow_pickle=True (with a console warning)
    Returns a dict mapping key->numpy array
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        # fallback (some npz contain pickled objects)
        try:
            print(f"[warn] np.load(..., allow_pickle=False) failed for {path} -> retrying with allow_pickle=True. Error: {e}")
            with np.load(path, allow_pickle=True) as d:
                return {k: d[k] for k in d.files}
        except Exception as e2:
            print(f"[error] failed to load npz even with allow_pickle=True: {e2}")
            raise

def summary_array(a: np.ndarray) -> Dict[str, Any]:
    """Return safe summary of an array. Works for numeric arrays; for object/non-numeric gives dtype & shape."""
    s: Dict[str, Any] = {}
    try:
        a_arr = np.asarray(a)
        s['shape'] = list(a_arr.shape)
        s['dtype'] = str(a_arr.dtype)
        # Numeric checks
        if np.issubdtype(a_arr.dtype, np.number):
            if a_arr.size:
                # use nan-aware functions
                s['min'] = float(np.nanmin(a_arr))
                s['max'] = float(np.nanmax(a_arr))
                s['mean'] = float(np.nanmean(a_arr))
                s['std'] = float(np.nanstd(a_arr))
                s['n_nan'] = int(np.isnan(a_arr).sum())
                s['n_inf'] = int(np.isinf(a_arr).sum())
            else:
                s.update({'min': None, 'max': None, 'mean': None, 'std': None, 'n_nan': 0, 'n_inf': 0})
        else:
            # non-numeric: provide sample values (truncated) and counts
            s['sample_preview'] = None
            try:
                flat = a_arr.ravel()
                take = min(5, flat.shape[0])
                s['sample_preview'] = [str(x) for x in flat[:take].tolist()]
            except Exception:
                s['sample_preview'] = str(type(a_arr.flat[0])) if a_arr.size else []
            s['n_nan'] = None
            s['n_inf'] = None
            s['min'] = None
            s['max'] = None
            s['mean'] = None
            s['std'] = None
    except Exception as e:
        s['error'] = str(e)
    return s

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_get_first(keys: List[str], container: Dict[str, Any]) -> Optional[str]:
    """Return first key from keys that exists in container (case-sensitive)."""
    for k in keys:
        if k in container:
            return k
    # case-insensitive search
    low_map = {kk.lower(): kk for kk in container.keys()}
    for k in keys:
        kk = k.lower()
        if kk in low_map:
            return low_map[kk]
    return None

# ---------------- main inspection ----------------
def inspect(out_dir: str, save_plots: bool = True, verbose: bool = False) -> Dict[str, Any]:
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"out_dir not found: {out_dir}")
    print(f"[info] Inspecting: {out_dir}")

    report: Dict[str, Any] = {"inspected_at": None, "meta": None, "npz": None, "raw_snapshot": None, "filter_sanity": None, "checks": [], "plots": []}
    import datetime
    report['inspected_at'] = datetime.datetime.now().isoformat()

    # load meta.json
    meta_path = os.path.join(out_dir, "meta.json")
    meta = load_json(meta_path)
    if meta is None:
        print("[warn] meta.json not found or failed to parse.")
    else:
        print("[info] meta.json loaded.")
        # keep the full meta for reference but also extract common fields
        report['meta'] = meta.copy()
        # extract important fields for quick report
        quick = {
            "class_map": meta.get("class_map"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_test": meta.get("n_test"),
            "class_counts": meta.get("class_counts"),
            "snr_list": meta.get("snr_list"),
            "generated_at": meta.get("generated_at"),
            "seed": meta.get("seed"),
        }
        report['meta_summary'] = quick

    # load filter_sanity.json
    fs_path = os.path.join(out_dir, "filter_sanity.json")
    fs = load_json(fs_path)
    if fs is None:
        print("[warn] filter_sanity.json not found or failed to parse.")
    else:
        print("[info] filter_sanity.json loaded.")
        report['filter_sanity'] = fs

    # find processed npz
    npz_path = find_npz(out_dir)
    data: Optional[Dict[str, np.ndarray]] = None
    if npz_path is None:
        print("[warn] No processed npz found in out_dir.")
    else:
        print(f"[info] Loading processed npz: {npz_path}")
        try:
            data = safe_load_npz(npz_path)
            report['npz_file'] = os.path.basename(npz_path)
            npz_summary: Dict[str, Any] = {}
            # examine keys & shapes
            for k, v in data.items():
                if verbose:
                    try:
                        print(f"  key: {k} -> shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
                    except Exception:
                        print(f"  key: {k} -> (couldn't print shape/dtype)")
                try:
                    npz_summary[k] = summary_array(v)
                except Exception as e:
                    npz_summary[k] = {"error": str(e)}
            report['npz'] = npz_summary
        except Exception as e:
            print(f"[error] failed to load/process npz: {e}")
            report['npz_error'] = str(e)
            data = None

    # load raw_snapshot.npz if present
    raw_snap_path = os.path.join(out_dir, "raw_snapshot.npz")
    raw: Optional[Dict[str, np.ndarray]] = None
    if os.path.exists(raw_snap_path):
        try:
            raw = safe_load_npz(raw_snap_path)
            print("[info] raw_snapshot.npz loaded.")
            report['raw_snapshot'] = {k: summary_array(v) for k, v in raw.items()}
        except Exception as e:
            print(f"[warn] failed to load raw_snapshot.npz: {e}")
    else:
        if verbose:
            print("[info] raw_snapshot.npz not found.")

    # quick checks: dimension consistency between meta and npz shapes
    checks = []
    try:
        meta_ntrain = None
        if report.get('meta'):
            meta_ntrain = report['meta'].get('n_train') or report.get('meta_summary', {}).get('n_train')
        xtrain_key = None
        if report.get('npz'):
            # check common X_train keys
            xtrain_key = safe_get_first(['X_train_sig', 'X_train', 'train_X', 'x_train', 'X_train_sig_ch0'], report['npz'])
        if xtrain_key:
            xtrain_shape = report['npz'][xtrain_key].get('shape')[0] if report['npz'][xtrain_key].get('shape') else None
            ok = (meta_ntrain == xtrain_shape) if (meta_ntrain is not None and xtrain_shape is not None) else None
            checks.append({"name": "n_train_matches_X_train", "meta_n_train": meta_ntrain, "X_train_n": xtrain_shape, "ok": ok})
    except Exception as e:
        checks.append({"name": "n_train_check_error", "error": str(e)})
    report['checks'] = checks

    # produce class counts figure if possible
    plots_dir = os.path.join(out_dir, "inspect_plots")
    ensure_dir(plots_dir)

    if HAS_MPL and save_plots:
        # Determine class counts (prefer meta.class_counts.train, else y_train in NPZ)
        class_counts = None
        try:
            if report.get('meta'):
                cc = report['meta'].get('class_counts') or report.get('meta_summary', {}).get('class_counts')
                # if class_counts is nested dict try to flatten for train
                if isinstance(cc, dict):
                    # possible structure: {'train': {'0':100,...}, 'test': {...}}
                    if 'train' in cc and isinstance(cc['train'], dict):
                        class_counts = {str(k): int(v) for k, v in cc['train'].items()}
                    else:
                        # try to interpret as direct counts mapping
                        try:
                            class_counts = {str(k): int(v) for k, v in cc.items()}
                        except Exception:
                            class_counts = None
            if class_counts is None and data is not None:
                # try many likely keys for y_train
                y_key = safe_get_first(['y_train', 'Y_train', 'train_y', 'y_train_labels'], data)
                if y_key:
                    y = np.asarray(data[y_key]).ravel()
                    unique, counts = np.unique(y, return_counts=True)
                    class_counts = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        except Exception as e:
            print(f"[warn] while trying to extract class counts: {e}")

        if class_counts:
            try:
                labels = list(class_counts.keys())
                vals = [class_counts[k] for k in labels]
                plt.figure(figsize=(8,4))
                plt.bar(labels, vals, color='tab:blue')
                plt.xlabel("Class (label or name)")
                plt.ylabel("n samples")
                plt.title("Per-class counts (train/test as available)")
                fn = os.path.join(plots_dir, "class_counts.png")
                plt.tight_layout()
                plt.savefig(fn, dpi=150)
                plt.close()
                report['plots'].append(os.path.basename(fn))
                print(f"[info] Saved class counts plot -> {fn}")
            except Exception as e:
                print(f"[warn] failed to generate class counts plot: {e}\n{traceback.format_exc()}")
        else:
            if verbose:
                print("[info] No class_counts found in meta or y_train not found; skipping class counts plot.")

        # plot top amplified bins per channel if filter_sanity exists
        if fs:
            try:
                # fs might have a top-level structure like {'ch0': {...}, 'ch1': {...}}
                for ch_key, ch_val in fs.items():
                    if not isinstance(ch_val, dict):
                        continue
                    top = ch_val.get('top_bins') or ch_val.get('top_bins_by_ratio') or ch_val.get('top_bins', [])
                    if not top or not isinstance(top, list):
                        continue
                    bins = []
                    rats = []
                    for item in top:
                        # item may be dict {'bin':..., 'ratio':...} or simple tuple
                        if isinstance(item, dict):
                            bins.append(int(item.get('bin', -1)))
                            rats.append(float(item.get('ratio', 0.0)))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            bins.append(int(item[0]))
                            rats.append(float(item[1]))
                    if not bins:
                        continue
                    plt.figure(figsize=(8,4))
                    plt.bar(bins, rats, color='tab:orange')
                    plt.xlabel("FFT bin index (rfft)")
                    plt.ylabel("amplification ratio (after/before)")
                    plt.title(f"Top amplified bins - {ch_key}")
                    fn = os.path.join(plots_dir, f"filter_top_bins_{ch_key}.png")
                    plt.tight_layout()
                    plt.savefig(fn, dpi=150)
                    plt.close()
                    report['plots'].append(os.path.basename(fn))
                    print(f"[info] Saved filter top bins plot -> {fn}")
            except Exception as e:
                print(f"[warn] failed to generate filter top bins plots: {e}\n{traceback.format_exc()}")
        else:
            if verbose:
                print("[info] No filter_sanity.json loaded; skipping top-bins plots.")

        # example spectrum before/after if raw snapshot present
        try:
            # try to find a raw signal key - common names: X_test_raw, X_raw, raw_snapshot, X_snapshot
            raw_sig = None
            if raw:
                raw_sig_key = safe_get_first(['X_test_raw', 'X_raw', 'raw', 'X_snapshot', 'X_raw_snapshot'], raw)
                if raw_sig_key:
                    raw_sig = np.asarray(raw[raw_sig_key])
            # alternative: sometimes processed npz contains a 'raw_snapshot' array
            if raw_sig is None and data is not None:
                alt_key = safe_get_first(['raw_snapshot', 'X_test_raw', 'X_raw', 'X_snapshot'], data)
                if alt_key:
                    raw_sig = np.asarray(data[alt_key])

            if raw_sig is not None and raw_sig.size:
                # pick first sample and channel (robust to shapes)
                sig = None
                if raw_sig.ndim == 3:
                    sig = raw_sig[0, 0, :]
                elif raw_sig.ndim == 2:
                    # (N, L)
                    sig = raw_sig[0, :]
                elif raw_sig.ndim == 1:
                    sig = raw_sig
                if sig is not None:
                    sig = np.asarray(sig).astype(float)
                    N = sig.shape[0]
                    freqs = np.fft.rfftfreq(N, d=1.0 / 12000)
                    spec = np.abs(np.fft.rfft(sig))
                    if HAS_MPL:
                        plt.figure(figsize=(8,4))
                        plt.semilogy(freqs, spec, label='raw snapshot (example)')
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Magnitude")
                        plt.title("Example raw snapshot spectrum (first sample)")
                        plt.xlim(0, freqs.max())
                        plt.legend()
                        fn = os.path.join(plots_dir, "example_spectrum_raw_before.png")
                        plt.tight_layout()
                        plt.savefig(fn, dpi=150)
                        plt.close()
                        report['plots'].append(os.path.basename(fn))
                        print(f"[info] Saved example spectrum (before) -> {fn}")

            else:
                if verbose:
                    print("[info] No raw snapshot signal found to make example spectrum.")
        except Exception as e:
            print(f"[warn] failed to create example spectrum: {e}\n{traceback.format_exc()}")

    else:
        if not HAS_MPL and save_plots:
            print("[warn] matplotlib not available, skipping plots.")
        elif not save_plots:
            if verbose:
                print("[info] save_plots disabled; skipping plots.")

    # final checks summary (NaNs / Infs / shape mismatches)
    sanity_checks = []
    try:
        if data:
            for k, v in data.items():
                try:
                    arr = np.asarray(v)
                    check = {'key': k, 'shape': list(arr.shape), 'dtype': str(arr.dtype)}
                    if np.issubdtype(arr.dtype, np.number):
                        check['n_nan'] = int(np.isnan(arr).sum())
                        check['n_inf'] = int(np.isinf(arr).sum())
                    else:
                        check['n_nan'] = None
                        check['n_inf'] = None
                    sanity_checks.append(check)
                except Exception as e:
                    sanity_checks.append({'key': k, 'error': str(e)})
        report['sanity_checks'] = sanity_checks
    except Exception as e:
        print(f"[warn] sanity checks failed: {e}")

    # save concise json report
    report_path = os.path.join(out_dir, "preprocess_report.json")
    try:
        with open(report_path, 'w', encoding='utf8') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"[info] Written concise report -> {report_path}")
    except Exception as e:
        print(f"[error] failed to write report json: {e}")

    return report

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Inspect & summarise outputs produced by Preprocess_refactored_v3_production.py"
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=False,                              # دیگر اجباری نیست
        default=r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust",
        help='Path to processed_data folder (containing meta.json, processed_data*.npz, etc.)'
    )
    
    parser.add_argument(
        '--no_plots',
        dest='save_plots',
        action='store_false',
        help='Do not save plots (only generate JSON report)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information (default: True)'
    )
    
    parser.add_argument(
        '--no_verbose',
        dest='verbose',
        action='store_false',
        help='Suppress detailed prints'
    )

    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    print(f"[info] Using out_dir: {out_dir}")

    try:
        inspect(out_dir, save_plots=args.save_plots, verbose=args.verbose)
        print("[success] Inspection completed successfully!")
    except Exception as e:
        print(f"[fatal] Inspection failed: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
