# -*- coding: utf-8 -*-
"""
visual_analytics_apex_fixed.py
Apex Analytics Suite - Final Fixed / Hardened Version
Author: Ali Naderi (adapted & fixed)
Purpose: Robust, production-ready visualization pipeline for CWRU processed npz outputs.
Features/fixes:
 - Robust NPZ key discovery and shape handling
 - Safe exports (kaleido failures handled)
 - Avoid broadcasting errors and handle missing keys gracefully
 - Balanced / sampled manifold plotting to avoid leakage / OOM
 - Informative logging
Usage:
    python visual_analytics_apex_fixed.py --data_dir "path/to/processed_data_robust"
"""
from __future__ import annotations
import os
import json
import logging
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch, hilbert, spectrogram
from scipy.stats import kurtosis, skew
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from PIL import Image

# ----------------- 0. Config / Logging -----------------
@dataclass
class Config:
    palette: List[str] = field(default_factory=lambda: [
        "#2980b9", "#c0392b", "#27ae60", "#f39c12",
        "#8e44ad", "#16a085", "#2c3e50", "#d35400",
        "#7f8c8d", "#2ecc71", "#e74c3c", "#6c5ce7"
    ])
    font: str = "Arial, Roboto, sans-serif"
    bg_color: str = "#ffffff"
    export_scale: float = 3.0  # used for PNG export scale
    default_fs: int = 12000
    max_tsne_per_class: int = 100
    radar_sample_per_class: int = 50
    stft_nperseg: int = 256
    envelope_nperseg: int = 2048
    image_width: int = 1600
    image_height: int = 900

CFG = Config()

# Create a light Plotly template for consistent look
pio.templates["ali_apex_fixed"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=CFG.font, color="#2d3436", size=13),
        paper_bgcolor=CFG.bg_color,
        plot_bgcolor="#f8f9fa",
        colorway=CFG.palette,
        hovermode="closest",
        title=dict(x=0.01, font=dict(size=20, family="Arial Black", color="#1a252f")),
        margin=dict(t=100, l=80, r=40, b=80),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#dfe6e9", borderwidth=1)
    )
)
pio.templates.default = "ali_apex_fixed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ApexAnalyticsFixed")


# ----------------- 1. Signal Physics Helpers -----------------
class SignalPhysics:
    @staticmethod
    def compute_8_metrics(signal: np.ndarray) -> Dict[str, float]:
        """Robust computation of 8 physics-based metrics. Accepts 1D signals."""
        s = np.asarray(signal).flatten()
        if s.size == 0:
            return {
                "RMS": 0.0, "Peak Value": 0.0, "Kurtosis": 0.0, "Skewness": 0.0,
                "Crest Factor": 0.0, "Shape Factor": 0.0, "Impulse Factor": 0.0, "Margin Factor": 0.0
            }
        abs_s = np.abs(s)
        rms = float(np.sqrt(np.mean(s ** 2)))
        peak = float(np.max(abs_s))
        try:
            kurt = float(kurtosis(s, fisher=False, bias=False))
        except Exception:
            kurt = 0.0
        try:
            skw = float(skew(s))
        except Exception:
            skw = 0.0

        denom = rms if rms > 0 else 1e-9
        mean_abs = np.mean(abs_s) if np.mean(abs_s) > 0 else 1e-9
        sqrt_mean = (np.mean(np.sqrt(abs_s))) ** 2 + 1e-9

        return {
            "RMS": rms,
            "Peak Value": peak,
            "Kurtosis": kurt,
            "Skewness": abs(skw),
            "Crest Factor": peak / denom,
            "Shape Factor": denom / mean_abs,
            "Impulse Factor": peak / mean_abs,
            "Margin Factor": peak / sqrt_mean
        }

    @staticmethod
    def get_envelope_spectrum(signal: np.ndarray, fs: int = None, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute envelope spectrum (frequency & PSD). nperseg is clamped to signal length."""
        if fs is None:
            fs = CFG.default_fs
        sig = np.asarray(signal).flatten()
        if sig.size == 0:
            return np.array([]), np.array([])
        analytic = hilbert(sig)
        amp_env = np.abs(analytic)
        amp_env = amp_env - np.mean(amp_env)
        if nperseg is None:
            nperseg = CFG.envelope_nperseg
        nperseg = min(nperseg, sig.size)
        if nperseg < 8:
            # fallback to a small power-of-two
            nperseg = max(8, sig.size // 2)
        try:
            f, p = welch(amp_env, fs=fs, nperseg=nperseg)
        except Exception:
            # final fallback simple FFT-based PSD estimate
            fft = np.fft.rfft(amp_env)
            p = np.abs(fft) ** 2
            f = np.fft.rfftfreq(len(amp_env), 1.0 / fs)
            # ensure same dtype
        return np.asarray(f), np.asarray(p)


# ----------------- 2. Utility: NPZ-safe access -----------------
def safe_get_npz(npz: Any, keys_candidates: List[str]) -> Optional[str]:
    """Return the first key in npz.files that matches any candidate (case-insensitive start or exact)."""
    if npz is None:
        return None
    files = [str(k) for k in getattr(npz, "files", [])]
    files_lower = [f.lower() for f in files]
    for cand in keys_candidates:
        cl = cand.lower()
        # exact name
        if cl in files_lower:
            return files[files_lower.index(cl)]
    # prefix startswith search
    for cand in keys_candidates:
        for f, fl in zip(files, files_lower):
            if fl.startswith(cand.lower()):
                return f
    # substring match
    for cand in keys_candidates:
        for f, fl in zip(files, files_lower):
            if cand.lower() in fl:
                return f
    return None


def safe_array(npz: Any, key: Optional[str]) -> Optional[np.ndarray]:
    if key is None: 
        return None
    try:
        arr = np.array(npz[key])
        return arr
    except Exception as e:
        logger.warning(f"Could not load array for key '{key}': {e}")
        return None


def sample_indices_by_class(y: np.ndarray, class_val: int, max_samples: int) -> np.ndarray:
    idxs = np.where(y == class_val)[0]
    if idxs.size == 0:
        return np.array([], dtype=int)
    if idxs.size <= max_samples:
        return idxs
    return np.random.choice(idxs, int(max_samples), replace=False)


# ----------------- 3. Main Visualizer -----------------
class ApexVisualizer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.report_dir = data_dir / "apex_report_final"
        self.assets_dir = self.report_dir / "assets"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # discover NPZ
        npz_candidates = sorted(list(self.data_dir.glob("*.npz")), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        if not npz_candidates:
            raise FileNotFoundError(f"No .npz files found under {self.data_dir}")
        self.npz_path = npz_candidates[0]
        logger.info(f"Using NPZ: {self.npz_path.name}")

        try:
            self.npz = np.load(self.npz_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load npz '{self.npz_path}': {e}")

        # meta
        self.meta = {}
        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse meta.json: {e}")
                self.meta = {}

        # mapping labels
        self.idx_to_label = self._parse_labels()
        # try to find y_train/y_test keys robustly
        y_train_key = safe_get_npz(self.npz, ["y_train", "train_y", "Y_train", "y_train_labels"])
        y_test_key = safe_get_npz(self.npz, ["y_test", "test_y", "Y_test", "y_test_labels"])
        self.y_train = safe_array(self.npz, y_train_key)
        self.y_test = safe_array(self.npz, y_test_key)

        # determine classes from y_train if possible else from mapped keys or meta
        if self.y_train is not None:
            self.all_classes = np.unique(self.y_train)
        elif self.y_test is not None:
            self.all_classes = np.unique(self.y_test)
        else:
            # try class map from meta
            raw_map = self.meta.get("idx_to_label") or self.meta.get("class_map") or {}
            if isinstance(raw_map, dict) and raw_map:
                try:
                    self.all_classes = np.array(sorted([int(k) for k in raw_map.keys()]))
                except Exception:
                    self.all_classes = np.arange(len(raw_map))
            else:
                self.all_classes = np.arange(10)  # fallback

        # friendly names
        self.class_names = [self.idx_to_label.get(int(c), str(int(c))) for c in self.all_classes]
        logger.info(f"Detected {len(self.all_classes)} classes: {self.class_names}")

        # Attempt to identify common array keys
        self.keys = list(getattr(self.npz, "files", []))
        # signal keys (sig) and feature keys (feat)
        self.X_train_sig_key = safe_get_npz(self.npz, ["X_train_sig", "x_train_sig", "X_train", "train_X", "x_train"])
        self.X_test_sig_key = safe_get_npz(self.npz, ["X_test_sig", "x_test_sig", "X_test", "test_X", "x_test"])
        self.X_train_feat_key = safe_get_npz(self.npz, ["X_train_feat", "x_train_feat", "X_train_features", "train_feat"])
        self.X_test_feat_key = safe_get_npz(self.npz, ["X_test_feat", "x_test_feat", "X_test_features", "test_feat"])

        # cache arrays (but keep None if not present)
        self.X_train_sig = safe_array(self.npz, self.X_train_sig_key)
        self.X_test_sig = safe_array(self.npz, self.X_test_sig_key)
        self.X_train_feat = safe_array(self.npz, self.X_train_feat_key)
        self.X_test_feat = safe_array(self.npz, self.X_test_feat_key)

        # channel names
        self.channels = self.meta.get("channels") or ["DE", "FE", "BA"]
        # safe sample rate
        self.fs = int(self.meta.get("fs", CFG.default_fs))

    def _parse_labels(self) -> Dict[int, str]:
        raw = self.meta.get("idx_to_label") or self.meta.get("class_map") or {}
        out = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    # fallback enumerated
                    out[len(out)] = str(v)
        if not out:
            # default placeholder
            out = {i: f"C{i}" for i in range(max(10, len(self.all_classes) if hasattr(self, "all_classes") else 10))}
        return out

    def _save_figure(self, fig: go.Figure, name: str, title: str = "", sub: str = "") -> str:
        """Robust save: HTML + try static image formats; returns basename (no ext)."""
        if title:
            full_title = f"<b>{title}</b><br><span style='font-size:13px;color:#7f8c8d'>{sub}</span>"
            try:
                fig.update_layout(title_text=full_title)
            except Exception:
                pass
        # footer
        try:
            fig.add_annotation(
                text="Ali Naderi | Apex Analytics", xref="paper", yref="paper",
                x=1, y=-0.12, showarrow=False, font=dict(size=10, color="#bdc3c7")
            )
        except Exception:
            pass

        base = self.assets_dir / name
        html_path = base.with_suffix(".html")
        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn", config={"displayModeBar": True})
        except Exception as e:
            logger.warning(f"Failed to write HTML for {name}: {e}")

        # static exports (try, but continue on failure)
        try:
            png_path = base.with_suffix(".png")
            svg_path = base.with_suffix(".svg")
            tif_path = base.with_suffix(".tif")
            # write_image may fail if kaleido not available; handle gracefully
            fig.write_image(str(png_path), scale=CFG.export_scale, width=CFG.image_width, height=CFG.image_height)
            try:
                fig.write_image(str(svg_path), width=CFG.image_width, height=CFG.image_height)
            except Exception:
                logger.debug("SVG export failed (non-critical).")
            # if PNG exists, create TIFF compressed
            if png_path.exists():
                try:
                    img = Image.open(str(png_path))
                    img.save(str(tif_path), compression="tiff_deflate")
                except Exception:
                    logger.debug("TIF export failed (non-critical).")
        except Exception as e:
            logger.debug(f"Static export failed for {name}: {e}")

        return name

    # ---------------- PLOTS ----------------

    def plot_dataset_hierarchy(self) -> Optional[str]:
        """Sunburst: Split -> Class -> count"""
        records = []
        for split in ["train", "val", "test"]:
            yk = safe_get_npz(self.npz, [f"y_{split}", f"y_{split}_labels", f"y_{split}"])
            if yk and yk in self.keys:
                y = safe_array(self.npz, yk)
                if y is None:
                    continue
                vals, cnts = np.unique(y, return_counts=True)
                for v, c in zip(vals, cnts):
                    records.append({"Split": split.upper(), "Class": self.idx_to_label.get(int(v), str(int(v))), "Count": int(c)})
        if not records:
            # fallback: try counts from y_train
            if self.y_train is None:
                logger.warning("No label arrays found for dataset hierarchy.")
                return None
            vals, cnts = np.unique(self.y_train, return_counts=True)
            for v, c in zip(vals, cnts):
                records.append({"Split": "TRAIN", "Class": self.idx_to_label.get(int(v), str(int(v))), "Count": int(c)})

        df = pd.DataFrame(records)
        fig = px.sunburst(df, path=['Split', 'Class'], values='Count', color='Split', color_discrete_sequence=CFG.palette)
        fig.update_layout(height=700, width=700)
        return self._save_figure(fig, "01_data_sunburst", "Dataset Composition", "Hierarchical view of samples")

    def plot_multi_sensor_radar(self) -> Optional[str]:
        """8-metric radar for each Class x Sensor (quantile-normalized)."""
        if self.X_test_sig is None or self.y_test is None:
            logger.warning("Missing X_test_sig or y_test for radar plot.")
            return None
        X = self.X_test_sig
        y = self.y_test

        # ensure shape (N, C, L) or (N, L)
        if X.ndim == 2:
            # single-channel features (treat as channel 0)
            X = X[:, np.newaxis, :]
        n_channels = X.shape[1]
        ch_names = self.meta.get("channels", self.channels)[:n_channels]

        metrics_rows = []
        for c in self.all_classes:
            idxs = sample_indices_by_class(y, int(c), CFG.radar_sample_per_class)
            if idxs.size == 0:
                continue
            for ch_idx in range(n_channels):
                # collect signals
                try:
                    sigs = X[idxs, ch_idx, :]
                except Exception:
                    # fallback to first dimension slicing
                    try:
                        sigs = X[idxs, :]
                    except Exception:
                        continue
                # compute metrics per sample
                batch = []
                for s in sigs:
                    try:
                        batch.append(SignalPhysics.compute_8_metrics(s))
                    except Exception:
                        batch.append(SignalPhysics.compute_8_metrics(np.zeros(1)))
                dfm = pd.DataFrame(batch).mean().to_dict()
                dfm["Class"] = self.idx_to_label.get(int(c), str(int(c)))
                dfm["Sensor"] = ch_names[ch_idx]
                metrics_rows.append(dfm)

        if not metrics_rows:
            logger.warning("No metrics computed for radar (empty).")
            return None

        df = pd.DataFrame(metrics_rows)
        numeric_cols = [c for c in df.columns if c not in ("Class", "Sensor")]
        if len(numeric_cols) == 0:
            logger.warning("No numeric metrics found for radar.")
            return None

        # Quantile transform to [0,1] for radar plotting
        qt = QuantileTransformer(output_distribution='uniform', random_state=0)
        try:
            df[numeric_cols] = qt.fit_transform(df[numeric_cols])
        except Exception:
            # fallback min-max per column
            for col in numeric_cols:
                colv = df[col].astype(float)
                mn, mx = colv.min(), colv.max()
                df[col] = (colv - mn) / (mx - mn + 1e-9)

        # Build polar traces: Class (Sensor) per trace
        fig = go.Figure()
        for _, row in df.iterrows():
            r = row[numeric_cols].astype(float).values
            theta = numeric_cols
            name = f"{row['Class']} ({row['Sensor']})"
            fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name=name, opacity=0.5))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=800, showlegend=True)
        return self._save_figure(fig, "02_radar_apex", "Multi-Sensor Statistical Fingerprint", "8 Physics Metrics (Quantile Normalized)")

    def plot_3d_envelope_waterfall_complete(self) -> Optional[str]:
        """3D envelope waterfall across all classes (uses first available sample per class)."""
        if self.X_train_sig is None or self.y_train is None:
            logger.warning("Missing X_train_sig or y_train for envelope waterfall.")
            return None
        X = self.X_train_sig
        y = self.y_train
        # ensure channel dimension
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        n_channels = X.shape[1]
        if n_channels == 0:
            logger.warning("No channels present in X_train_sig.")
            return None

        # select DE channel index if available
        de_idx = 0
        # try to find index for 'DE' in channels meta
        for i, ch in enumerate(self.channels):
            if "DE" in ch.upper() or "DRIVE" in ch.upper():
                de_idx = i
                break
        # sorted classes by label name for stable ordering
        sorted_classes = sorted(self.all_classes, key=lambda c: str(self.idx_to_label.get(int(c), str(int(c)))))

        fig = go.Figure()
        for i, c in enumerate(sorted_classes):
            idxs = np.where(y == c)[0]
            if idxs.size == 0:
                continue
            idx = int(idxs[0])
            try:
                sig = X[idx, de_idx, :]
            except Exception:
                try:
                    sig = X[idx, 0, :]
                except Exception:
                    continue
            f, p = SignalPhysics.get_envelope_spectrum(sig, fs=self.fs, nperseg=min(CFG.envelope_nperseg, sig.size))
            if f.size == 0 or p.size == 0:
                continue
            p_db = 10.0 * np.log10(p + 1e-12)
            # limit frequency range for plotting to reasonable band if too wide
            mask = (f > 10) & (f < min(5000, self.fs/2))
            if not np.any(mask):
                mask = slice(None)
            lbl = self.idx_to_label.get(int(c), str(int(c)))
            color = CFG.palette[i % len(CFG.palette)]
            # y coordinate is class index for stacking
            yvals = np.full_like(f[mask], fill_value=i, dtype=float)
            fig.add_trace(go.Scatter3d(x=f[mask], y=yvals, z=p_db[mask], mode='lines', line=dict(width=4, color=color), name=lbl,
                                      hovertemplate=f"<b>{lbl}</b><br>Freq: %{{x:.1f}}Hz<br>Amp: %{{z:.1f}}dB"))
        fig.update_layout(height=900,
                          scene=dict(
                              xaxis=dict(title="Frequency (Hz)"),
                              yaxis=dict(title="Class Index", tickvals=list(range(len(sorted_classes))),
                                         ticktext=[self.idx_to_label.get(int(c), "") for c in sorted_classes]),
                              zaxis=dict(title="Envelope Power (dB)"),
                              camera=dict(eye=dict(x=1.7, y=-1.7, z=0.7)),
                              aspectratio=dict(x=1, y=1.5, z=0.7)
                          ))
        return self._save_figure(fig, "03_envelope_waterfall", "3D Envelope Spectrum Waterfall", "Envelope spectral signature for classes")

    def plot_stft_surface_grid(self) -> Optional[str]:
        """STFT surface grid: per-class STFT surface plotted in a subplot grid."""
        if self.X_train_sig is None or self.y_train is None:
            logger.warning("Missing signals for STFT grid.")
            return None
        X = self.X_train_sig
        y = self.y_train
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        n_channels = X.shape[1]
        ch_idx = 0  # choose channel 0 for STFT grid by default
        sorted_classes = sorted(self.all_classes, key=lambda c: str(self.idx_to_label.get(int(c), str(int(c)))))
        if len(sorted_classes) == 0:
            return None
        cols = 3
        rows = (len(sorted_classes) + cols - 1) // cols
        specs = [[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)]
        titles = [self.idx_to_label.get(int(c), str(int(c))) for c in sorted_classes]
        fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.06)

        for i, c in enumerate(sorted_classes):
            idxs = np.where(y == c)[0]
            if idxs.size == 0:
                continue
            idx = int(idxs[0])
            try:
                sig = X[idx, ch_idx, :]
            except Exception:
                continue
            # choose nperseg clamped to length
            nperseg = min(CFG.stft_nperseg, sig.size)
            if nperseg < 8:
                nperseg = max(8, sig.size // 2)
            try:
                f, t, Sxx = spectrogram(sig, fs=self.fs, nperseg=nperseg)
            except Exception as e:
                logger.debug(f"Spectrogram failed for class {c}: {e}")
                continue
            if Sxx.size == 0:
                continue
            Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
            r = (i // cols) + 1
            c_idx = (i % cols) + 1
            # Plotly Surface expects z as 2D (len(y) x len(x)) matching f x t
            try:
                surf = go.Surface(z=Sxx_db, x=t, y=f, colorscale='Viridis', showscale=False)
                fig.add_trace(surf, row=r, col=c_idx)
            except Exception as e:
                logger.debug(f"Adding surface failed for class {c}: {e}")
                continue

        fig.update_layout(height=300 * rows, title_text="3D STFT Time-Frequency Topography")
        return self._save_figure(fig, "04_stft_grid", "Time-Frequency Topography", "STFT Surface Plots for All Classes")

    def plot_parallel_feature_flow(self) -> Optional[str]:
        """Parallel coordinates for test features (subsampled)."""
        if self.X_test_feat is None or self.y_test is None:
            logger.warning("Missing test features for parallel coordinates.")
            return None
        X = self.X_test_feat
        y = self.y_test
        # ensure 2D (N, F)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_features = X.shape[1]
        take = min(8, n_features)
        n_samples = min(500, X.shape[0])
        if n_samples == 0:
            return None
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sub = X[idx, :take]
        y_sub = y[idx]
        df = pd.DataFrame(X_sub, columns=[f"F{i}" for i in range(take)])
        df["Class"] = [self.idx_to_label.get(int(v), str(int(v))) for v in y_sub]
        try:
            fig = px.parallel_coordinates(df, color=df["Class"].astype('category').cat.codes,
                                          labels={c: c for c in df.columns}, color_continuous_scale=px.colors.diverging.Tealrose)
            fig.update_layout(height=500)
            return self._save_figure(fig, "05_parallel_coords", "Feature Decision Flow", "Parallel coordinates (sampled)")
        except Exception as e:
            logger.warning(f"Parallel coordinates generation failed: {e}")
            return None

    def plot_tsne_manifold(self) -> Optional[str]:
        """t-SNE manifold (2D) sampled per class to avoid imbalance and OOM."""
        if self.X_train_feat is None or self.y_train is None:
            logger.warning("Missing features for t-SNE manifold.")
            return None
        X = self.X_train_feat
        y = self.y_train
        # flatten features if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        # sample up to CFG.max_tsne_per_class per class
        indices = []
        for c in self.all_classes:
            idxs = np.where(y == c)[0]
            if idxs.size == 0:
                continue
            k = min(CFG.max_tsne_per_class, idxs.size)
            chosen = np.random.choice(idxs, k, replace=False)
            indices.extend(list(chosen))
        if len(indices) < 2:
            logger.warning("Not enough samples for t-SNE.")
            return None
        X_sub = X[indices]
        y_sub = y[indices]
        # run t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(indices)//3)))
            X_emb = tsne.fit_transform(X_sub)
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")
            return None
        labels = [self.idx_to_label.get(int(c), str(int(c))) for c in y_sub]
        fig = px.scatter(x=X_emb[:, 0], y=X_emb[:, 1], color=labels, color_discrete_sequence=CFG.palette, opacity=0.8)
        fig.update_traces(marker=dict(size=7, line=dict(width=1, color='white')))
        fig.update_layout(height=800, xaxis_title="Dim 1", yaxis_title="Dim 2")
        return self._save_figure(fig, "06_tsne_manifold", "t-SNE Feature Cluster", "Global class separability map (sampled)")

    def plot_robustness_curves(self) -> Optional[str]:
        """Feature drift vs SNR using available test_snr_*_feat keys."""
        snr_list = self.meta.get("snr_list") or []
        if not snr_list:
            # try to infer from npz keys
            inferred = []
            for k in self.keys:
                m = None
                # match patterns like test_snr_-10_feat or test_snr_10_feat etc.
                if "test_snr_" in k and ("_feat" in k or "feat" in k):
                    # extract digits and sign
                    import re
                    mm = re.search(r"test_snr_(-?\d+)", k)
                    if mm:
                        try:
                            inferred.append(int(mm.group(1)))
                        except Exception:
                            pass
            snr_list = sorted(set(inferred))
        if not snr_list:
            logger.warning("No SNR info for robustness curves.")
            return None
        if self.X_test_feat is None or self.y_test is None:
            logger.warning("Missing test features for robustness curves.")
            return None

        fig = go.Figure()
        base_key = self.X_test_feat_key
        if base_key is None or base_key not in self.keys:
            # try common base
            base_key = safe_get_npz(self.npz, ["X_test_feat", "x_test_feat", "test_feat", "X_test_features"])
        ref_feat = safe_array(self.npz, base_key)
        y_test = self.y_test
        if ref_feat is None:
            logger.warning("Reference features not found for robustness curves.")
            return None

        for c in self.all_classes:
            idxs = np.where(y_test == c)[0]
            if idxs.size == 0:
                continue
            # mean of reference features for this class
            try:
                ref_c = np.mean(ref_feat[idxs], axis=0)
            except Exception:
                # fallback flatten
                try:
                    ref_c = np.mean(ref_feat[idxs].reshape(len(idxs), -1), axis=0)
                except Exception:
                    continue
            drifts = []
            valid_snrs = []
            for snr in sorted(snr_list):
                key = safe_get_npz(self.npz, [f"test_snr_{snr}_feat", f"test_snr_{snr}_features"])
                if key and key in self.keys:
                    curr = safe_array(self.npz, key)
                    if curr is None:
                        continue
                    try:
                        curr_c = np.mean(curr[idxs], axis=0)
                    except Exception:
                        try:
                            curr_c = np.mean(curr[idxs].reshape(len(idxs), -1), axis=0)
                        except Exception:
                            continue
                    # compute Euclidean drift (flatten if needed)
                    try:
                        dist = float(np.linalg.norm(ref_c.flatten() - curr_c.flatten()))
                    except Exception:
                        dist = float(np.linalg.norm(np.asarray(ref_c).ravel() - np.asarray(curr_c).ravel()))
                    drifts.append(dist)
                    valid_snrs.append(snr)
            if not valid_snrs:
                continue
            fig.add_trace(go.Scatter(x=valid_snrs, y=drifts, mode='lines+markers', name=self.idx_to_label.get(int(c), str(int(c)))))
        fig.update_layout(xaxis_title="SNR (dB)", yaxis_title="Feature Drift (Euclidean)", height=600)
        return self._save_figure(fig, "07_robustness", "Robustness Analysis", "Feature stability vs noise levels")

    # ---------------- DASHBOARD ----------------
    def build_dashboard(self, plots: List[Tuple[str, str]]):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Apex Analytics | CWRU</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;600&display=swap" rel="stylesheet">
    <style>
        :root {{ --bg: #f4f7f6; --card: #ffffff; --text: #2c3e50; --primary: #2980b9; }}
        body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }}
        .header {{ text-align: center; margin-bottom: 3rem; }}
        .header h1 {{ font-weight: 800; font-size: 2.5rem; letter-spacing: -1px; margin-bottom: 0.5rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 2rem; }}
        .card {{ background: var(--card); border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); overflow: hidden; }}
        .card-head {{ padding: 1rem; background: #fbfbfb; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
        .card-title {{ font-weight: 700; font-size: 1rem; }}
        .btn {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #e0e0e0; text-decoration: none; color: #666; font-size: 0.85rem; transition: 0.2s; margin-left: 5px; }}
        .btn:hover {{ background: var(--primary); color: white; border-color: var(--primary); }}
        iframe {{ width: 100%; height: 650px; border: none; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>APEX ANALYTICS</h1>
        <p>Comprehensive Diagnosis for {len(self.all_classes)} Bearing Fault Classes</p>
    </div>
    <div class="grid">
"""
        for title, fname in plots:
            if not fname:
                continue
            html += f"""
        <div class="card">
            <div class="card-head">
                <span class="card-title">{title}</span>
                <div>
                    <a href="assets/{fname}.png" class="btn" download>PNG</a>
                    <a href="assets/{fname}.svg" class="btn" download>SVG</a>
                    <a href="assets/{fname}.tif" class="btn" download>TIF</a>
                </div>
            </div>
            <iframe src="assets/{fname}.html" loading="lazy"></iframe>
        </div>"""
        html += """
    </div>
</body>
</html>
"""
        try:
            with open(self.report_dir / "index.html", "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Report written: {self.report_dir / 'index.html'}")
        except Exception as e:
            logger.error(f"Failed to write dashboard index.html: {e}")


# ----------------- Main CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp", help="Data Path")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)

    viz = ApexVisualizer(data_path)

    tasks = [
        ("Dataset Hierarchy", viz.plot_dataset_hierarchy),
        ("3D Envelope Waterfall (All Classes)", viz.plot_3d_envelope_waterfall_complete),
        ("8-Metric Physics Radar", viz.plot_multi_sensor_radar),
        ("Feature Flow (Parallel Coords)", viz.plot_parallel_feature_flow),
        ("3D STFT Surface Grid", viz.plot_stft_surface_grid),
        ("Global T-SNE Manifold", viz.plot_tsne_manifold),
        ("Noise Robustness Curves", viz.plot_robustness_curves)
    ]

    results: List[Tuple[str, str]] = []
    for name, func in tasks:
        logger.info(f"Generating: {name} ...")
        try:
            out = func()
            if out:
                results.append((name, out))
                logger.info(f"Generated: {name} -> {out}")
            else:
                logger.info(f"Skipped (no output): {name}")
        except Exception as e:
            logger.exception(f"Failed to generate '{name}': {e}")

    viz.build_dashboard(results)
    logger.info("Apex Pipeline Complete. Report available at: %s", viz.report_dir / "index.html")

if __name__ == "__main__":
    main()
