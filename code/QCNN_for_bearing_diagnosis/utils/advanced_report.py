# -*- coding: utf-8 -*-
"""
advanced_report_paper.py
Author: Ali Naderi - 2025 (Journal/Paper Ready)
Purpose: Generate Publication-Quality Statistical & Physics Reports for Bearings

Notes:
 - Hilbert / HHT usage removed (per request). Envelope-based analysis replaced with
   power spectral density (PSD) via Welch on raw/detrended signals.
 - This script is robust to missing keys and uses a safe NPZ loader (tries allow_pickle=False, then True).
 - Produces interactive HTML + static PNG/PDF exports where possible.
"""
from __future__ import annotations
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import welch
import plotly.io as pio
import time

# ------------------------- Settings -------------------------
pio.templates.default = "plotly_white"

PRO_PALETTE = [
    "#2C3E50", "#E74C3C", "#F1C40F", "#27AE60",
    "#8E44AD", "#E67E22", "#16A085", "#34495E"
]

LAYOUT_SETTINGS = dict(
    font=dict(family="Arial, Roboto, Helvetica", size=14, color="#333"),
    title=dict(font=dict(size=20, family="Arial", color="#111")),
    legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1),
    margin=dict(l=80, r=40, t=80, b=80),
    xaxis=dict(showgrid=True, gridcolor='#EAEAEA', zeroline=True, linecolor='#333', linewidth=1.0),
    yaxis=dict(showgrid=True, gridcolor='#EAEAEA', zeroline=True, linecolor='#333', linewidth=1.0),
    plot_bgcolor="white",
    paper_bgcolor="white",
    colorway=PRO_PALETTE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("AdvReportPaper")

DEFAULT_DATA_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp"
NPZ_NAME = "processed_data_robust_v3.npz"  # default name; if missing, try to discover
META_NAME = "meta.json"
REPORT_DIR_NAME = "paper_ready_report"
IMAGES_SUBDIR = "images"

# ------------------------- Helpers -------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def safe_load_npz(path: Path) -> Dict[str, np.ndarray]:
    """
    Load an .npz file robustly:
      - first tries allow_pickle=False
      - fallback to allow_pickle=True with a warning
    Returns dict of arrays.
    """
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    try:
        with np.load(str(path), allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        logger.warning(f"np.load(..., allow_pickle=False) failed for {path}: {e} — retrying with allow_pickle=True")
        try:
            with np.load(str(path), allow_pickle=True) as d:
                return {k: d[k] for k in d.files}
        except Exception as e2:
            logger.error(f"Failed to load NPZ even with allow_pickle=True: {e2}")
            raise

def find_npz_file(data_dir: Path, candidate: Optional[str] = None) -> Optional[Path]:
    """Find the correct processed_data_*.npz file — smart & robust"""
    if candidate:
        p = data_dir / candidate
        if p.exists():
            return p

    # فقط فایل‌هایی که با processed_data_ شروع میشن رو در نظر بگیر
    candidates = list(data_dir.glob("processed_data_*.npz"))
    
    if not candidates:
        # اگر هیچی نبود، حداقل یک npz پیدا کن (برای خطای بهتر)
        fallback = list(data_dir.glob("*.npz"))
        if fallback:
            logger.warning(f"No 'processed_data_*.npz' found! Found these instead: {[p.name for p in fallback]}")
            # return the latest fallback
            return max(fallback, key=lambda p: p.stat().st_mtime)
        return None

    # آخرین فایل از نظر تاریخ ساخت (نه الفبایی!)
    latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Found {len(candidates)} processed_data_*.npz files. Using latest by modification time:")
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
        logger.info(f"   → {p.name} (modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))})")
    
    return latest_file

def safe_get(arrs: Dict[str, np.ndarray], keys: List[str]) -> Optional[str]:
    """Return first existing key from keys (case-sensitive then case-insensitive)."""
    for k in keys:
        if k in arrs:
            return k
    low_map = {kk.lower(): kk for kk in arrs.keys()}
    for k in keys:
        kk = k.lower()
        if kk in low_map:
            return low_map[kk]
    return None

def save_figure(fig: go.Figure, out_dir: Path, name: str, w: int = 1400, h: int = 900) -> Path:
    """Save interactive HTML and try static PNG/PDF exports; returns the HTML path."""
    fig.update_layout(**LAYOUT_SETTINGS)
    html_path = out_dir / f"{name}.html"
    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    except Exception as e:
        logger.warning(f"Failed to write HTML for {name}: {e}")
    # try static
    try:
        png_path = out_dir / f"{name}.png"
        pdf_path = out_dir / f"{name}.pdf"
        # scale=2 for higher-res PNG
        fig.write_image(str(png_path), width=w, height=h, scale=2)
        fig.write_image(str(pdf_path), width=w, height=h)
    except Exception as e:
        logger.warning(f"Static export failed for {name}: {e}")
    return html_path

def class_map_from_meta(meta: Dict[str, Any]) -> Dict[int, str]:
    """
    Normalise class_map shapes:
      - meta['class_map'] might be {'0': 'Normal', ...} or { 'Normal': 0 }
      - return mapping int->name
    """
    if not meta:
        return {}
    cm = meta.get("class_map") or meta.get("idx_to_label") or meta.get("label_map")
    if not cm:
        return {}
    # if mapping is name->idx, invert it
    ints = {}
    try:
        # detect if values are ints (then keys are names)
        first_val = next(iter(cm.values()))
        if isinstance(first_val, int):
            for name, idx in cm.items():
                ints[int(idx)] = str(name)
            return ints
        else:
            # keys maybe strings of ints
            for k, v in cm.items():
                try:
                    ints[int(k)] = str(v)
                except Exception:
                    # fallback: sequential numbering
                    pass
            if ints:
                return ints
    except Exception:
        pass
    # final fallback: try cast keys to int
    try:
        result = {int(k): str(v) for k, v in cm.items()}
        return result
    except Exception:
        # last resort: create enumerated mapping
        return {i: f"C{i}" for i in range(len(cm))}

# ------------------------- Load dataset -------------------------
def load_dataset(data_dir: Path, npz_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns a dict containing arrays and meta. Safe to call even when some keys are missing.
    """
    npz_path = find_npz_file(data_dir, npz_name)
    if npz_path is None:
        raise FileNotFoundError(f"No NPZ file found in {data_dir}")
    logger.info(f"Using NPZ: {npz_path}")
    arrays = safe_load_npz(npz_path)

    meta_path = data_dir / META_NAME
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read meta.json: {e}")

    class_map = class_map_from_meta(meta)
    # fallback class_map from y keys
    if not class_map:
        # try infer labels from y_train unique
        y_key = safe_get(arrays, ["y_train", "y_train_labels", "Y_train", "y"])
        if y_key:
            try:
                y = np.asarray(arrays[y_key]).ravel()
                uniques = np.unique(y)
                class_map = {int(u): f"C{int(u)}" for u in uniques}
            except Exception:
                class_map = {}

    # safely extract common keys (if missing, leave None)
    def _get(klist):
        key = safe_get(arrays, klist)
        return arrays[key] if key else None

    data = {
        "X_train_sig": _get(["X_train_sig", "X_train", "train_X", "X_train_raw"]),
        "X_train_feat": _get(["X_train_feat", "X_feat", "X_train_features"]),
        "y_train": _get(["y_train", "Y_train", "train_y"]),
        "X_val_sig": _get(["X_val_sig", "X_val", "val_X"]),
        "X_val_feat": _get(["X_val_feat", "X_val_features"]),
        "y_val": _get(["y_val", "Y_val", "val_y"]),
        "X_test_sig": _get(["X_test_sig", "X_test", "test_X"]),
        "X_test_feat": _get(["X_test_feat", "X_test_features"]),
        "y_test": _get(["y_test", "Y_test", "test_y"]),
        "class_map": class_map,
        "meta": meta,
        "npz_path": npz_path
    }
    return data

# ------------------------- Analysis Functions -------------------------
def plot_pca_clusters(data: Dict[str, Any], out_dir: Path) -> Tuple[str, Path]:
    logger.info("Plotting PCA clusters...")
    X_feat = data.get("X_train_feat")
    y = data.get("y_train")
    class_map = data.get("class_map", {})

    if X_feat is None:
        raise ValueError("X_train_feat not found in dataset.")
    if y is None:
        raise ValueError("y_train not found in dataset.")

    X = np.asarray(X_feat)
    y = np.asarray(y).ravel()

    # Standardize then PCA
    scaler = StandardScaler()
    try:
        X_norm = scaler.fit_transform(X)
    except Exception:
        # fallback: try column-wise scaling
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    pca = PCA(n_components=min(3, X_norm.shape[1]))
    X_pca = pca.fit_transform(X_norm)
    var_explained = np.sum(pca.explained_variance_ratio_) * 100

    df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    df["Class"] = [class_map.get(int(l), str(int(l))) for l in y]

    fig = px.scatter(df, x="PC1", y="PC2", color="Class",
                     title=f"PCA Projection (Explained Variance {var_explained:.1f}%)",
                     color_discrete_sequence=PRO_PALETTE, opacity=0.85)
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color='white')))
    out = save_figure(fig, out_dir, "pca_clusters")
    return ("PCA Clusters", out)

def plot_envelope_spectrum(data: Dict[str, Any], out_dir: Path) -> Tuple[str, Path]:
    """
    Note: 'Envelope' term kept for API compatibility, but implementation uses PSD (Welch)
    on raw/detrended signals because Hilbert/HHT was NOT used in preprocess.
    """
    logger.info("Plotting power spectral density (PSD) per class using Welch (no Hilbert).")
    X = data.get("X_train_sig")
    y = data.get("y_train")
    class_map = data.get("class_map", {})

    if X is None or y is None:
        raise ValueError("X_train_sig or y_train missing for spectrum analysis.")

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    unique_classes = np.unique(y)
    n = len(unique_classes)
    # limit subplot rows to a reasonable number for readability
    max_rows = min(n, 12)
    titles = [class_map.get(int(c), str(int(c))) for c in unique_classes[:max_rows]]
    fig = make_subplots(rows=max_rows, cols=1, shared_xaxes=True, subplot_titles=titles)

    for i, cid in enumerate(unique_classes[:max_rows]):
        idxs = np.where(y == cid)[0]
        if idxs.size == 0:
            continue
        idx = int(idxs[0])
        # robust indexing for channels/length
        sig = None
        try:
            if X.ndim == 3:
                # assume shape (N, channels, samples)
                sig = X[idx, 0, :]
            elif X.ndim == 2:
                sig = X[idx, :]
            else:
                sig = np.ravel(X)[idx]
        except Exception:
            # last resort: try flatten and slice a chunk
            arr = np.asarray(X)
            arr_flat = arr.reshape(-1)
            start = min(idx * 1024, max(0, arr_flat.size - 4096))
            sig = arr_flat[start:start + 4096]

        sig = np.asarray(sig).astype(float)
        # detrend simple mean to avoid DC
        sig = sig - np.mean(sig)
        # compute PSD using Welch
        nperseg = min(2048, max(256, sig.shape[0] // 4))
        try:
            f, Pxx = welch(sig, fs=12000, nperseg=nperseg)
        except Exception:
            # fallback simpler FFT-based PSD
            fft = np.abs(np.fft.rfft(sig))**2
            f = np.fft.rfftfreq(sig.shape[0], d=1/12000)
            Pxx = fft

        # limit to <1000 Hz for plotting clarity
        mask = f < 1000
        trace = go.Scatter(x=f[mask], y=Pxx[mask], line=dict(color=PRO_PALETTE[i % len(PRO_PALETTE)], width=1.5),
                           name=class_map.get(int(cid), str(int(cid))))
        fig.add_trace(trace, row=i+1, col=1)

    fig.update_layout(height=220 * max_rows, title="Power Spectral Density (Welch) per-class sample (no Hilbert)")
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="PSD / Power")
    out = save_figure(fig, out_dir, "power_spectrum")
    return ("Power Spectrum (PSD)", out)

def plot_violin_features(data: Dict[str, Any], out_dir: Path) -> Tuple[str, Path]:
    logger.info("Plotting violin feature distribution...")
    X_feat = data.get("X_train_feat")
    y = data.get("y_train")
    class_map = data.get("class_map", {})

    if X_feat is None or y is None:
        raise ValueError("Required X_train_feat or y_train missing for violin plot.")

    X = np.asarray(X_feat)
    y = np.asarray(y).ravel()

    # create dataframe for first few features (avoid overplotting)
    n_features = min(X.shape[1], 6)
    cols = [f"Feat{i}" for i in range(n_features)]
    df = pd.DataFrame(X[:, :n_features], columns=cols)
    df["Class"] = [class_map.get(int(l), str(int(l))) for l in y]

    # make a combined violin for one representative feature (Feat0) and a small grid for others
    fig = px.violin(df, x="Class", y="Feat0", color="Class", box=True, points="outliers",
                    title="Feature Distribution by Class (Violin) - Feature 0")
    out = save_figure(fig, out_dir, "violin_feature0")
    # additionally create small grid of boxplots for first n_features
    fig2 = make_subplots(rows=1, cols=n_features, subplot_titles=cols)
    for i, col in enumerate(cols):
        fig2.add_trace(go.Box(y=df[col], name=col, boxmean='sd'), row=1, col=i+1)
    fig2.update_layout(height=350, title="Feature summaries (first features)")
    out2 = save_figure(fig2, out_dir, "feature_summaries")
    return ("Violin Features", out)

def plot_channel_corr(data: Dict[str, Any], out_dir: Path) -> Tuple[str, Path]:
    logger.info("Plotting channel correlation matrix...")
    X_feat = data.get("X_train_feat")
    if X_feat is None:
        raise ValueError("X_train_feat missing for channel correlation.")
    X = np.asarray(X_feat)
    # if features correspond to channels, attempt correlation across columns
    corr = np.corrcoef(X.T)
    channels = [f"Ch{i}" for i in range(corr.shape[0])]
    fig = px.imshow(corr, x=channels, y=channels, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", title="Feature/Channel Correlation Matrix")
    out = save_figure(fig, out_dir, "channel_correlation")
    return ("Channel Correlations", out)

# ------------------------- Dashboard -------------------------
def generate_dashboard(out_dir: Path, plots: List[Tuple[str, Path]]):
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Ready Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; background:#f7f8fb; color:#222; }}
    h1 {{ font-size:28px; margin-bottom:6px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(720px,1fr)); gap:16px; }}
    .card {{ background:white; border-radius:8px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
    iframe {{ width:100%; height:620px; border:none; }}
    .meta {{ font-size:13px; color:#666; margin-bottom:10px; }}
  </style>
</head>
<body>
<h1>Scientific Bearing Report (Paper-ready)</h1>
<p class="meta">Generated by advanced_report_paper.py</p>
<div class="grid">
"""
    for title, path in plots:
        if not path:
            continue
        name = path.stem
        html += f"""<div class="card"><h3>{title}</h3><iframe src="{IMAGES_SUBDIR}/{name}.html"></iframe></div>"""
    html += "</div></body></html>"
    idx = out_dir / "index.html"
    with open(idx, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard written to {idx}")

# ------------------------- Main -------------------------
def main(args):
    base_dir = Path(args.base_dir).expanduser().resolve()
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    logger.info(f"Loading dataset from: {base_dir}")

    # انتخاب فایل NPZ هوشمند:
    npz_path: Optional[Path] = None

    if args.npz_name:
        candidate = base_dir / args.npz_name
        if candidate.exists():
            npz_path = candidate
            logger.info(f"Using explicitly specified NPZ: {npz_path.name}")
        else:
            logger.warning(f"Specified NPZ not found: {args.npz_name} → falling back to auto-discovery")
    
    if npz_path is None:
        npz_path = find_npz_file(base_dir)
        if npz_path is None:
            raise FileNotFoundError(
                f"No processed_data_*.npz file found in {base_dir}\n"
                "   موجود نیست! فایل‌های موجود:\n   " + 
                "\n   ".join([p.name for p in base_dir.glob("*.npz")][:10]) or "هیچ .npz ای پیدا نشد"
            )
        logger.info(f"Auto-discovered latest NPZ: {npz_path.name}")

    # لود دیتاست با فایل درست
    data = load_dataset(base_dir, npz_name=npz_path.name)  # فقط اسم فایل رو بده

    # ساخت پوشه‌های خروجی
    out_dir = base_dir / REPORT_DIR_NAME
    img_dir = out_dir / IMAGES_SUBDIR
    ensure_dir(img_dir)

    logger.info(f"Report will be saved to: {out_dir}")

    plot_registry: List[Tuple[str, Path]] = []

    # توابع رسم نمودار
    plot_functions = [
        ("PCA Clusters", plot_pca_clusters),
        ("Power Spectrum (PSD) per-class", plot_envelope_spectrum),
        ("Feature Distribution (Violin/Box)", plot_violin_features),
        ("Feature Correlation Matrix", plot_channel_corr),
    ]

    for title, func in plot_functions:
        try:
            logger.info(f"Generating: {title} ...")
            saved = func(data, img_dir)
            plot_registry.append(saved)
            logger.info(f"Success: {title} → {saved[1].name}")
        except Exception as e:
            logger.error(f"Failed: {title} | Error: {e}", exc_info=False)

    # ساخت داشبورد نهایی
    generate_dashboard(out_dir, plot_registry)
    
    index_path = out_dir / "index.html"
    logger.info(f"Paper-ready report generated successfully!")
    logger.info(f"Open this file in browser: {index_path}")
    logger.info(f"Direct link: file://{index_path}")

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="advanced_report_paper.py",
        description="Generate Publication-Quality Interactive + Static Report for CWRU Bearing Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=DEFAULT_DATA_PATH,
        help="Folder containing processed_data_*.npz and meta.json (e.g. .../3HP/processed_data_robust)"
    )
    
    parser.add_argument(
        "--npz_name", 
        type=str, 
        default=None,
        help="Optional: exact name of NPZ file (e.g. processed_data_robust_v3.npz). If not given, latest is used."
    )
    
    parser.add_argument(
        "--open", 
        action="store_true", 
        help="Automatically open the report in default browser after generation"
    )

    args = parser.parse_args()

    try:
        main(args)
        
        # اختیاری: باز کردن خودکار مرورگر
        if args.open:
            import webbrowser
            index_file = Path(args.base_dir) / REPORT_DIR_NAME / "index.html"
            if index_file.exists():
                webbrowser.open(f"file://{index_file.resolve()}")
                logger.info("Report opened in browser!")
            else:
                logger.warning("index.html not found.")
                
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        raise
    finally:
        print("\n" + "="*60)
        print("   Paper-Ready Report Generation Finished!")
        print("="*60 + "\n")

