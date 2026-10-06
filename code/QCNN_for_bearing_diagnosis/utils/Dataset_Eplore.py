# # -*- coding: utf-8 -*-
# """
# inspect_cwru_files_professional.py
# Author: Ali Naderi - 2025 (Refined & Professional)
# Purpose: Comprehensive Inspection of CWRU Dataset Files
# Features:
#   - Robust MATLAB file parsing (handles various formats)
#   - Intelligent channel identification (DE, FE, BA)
#   - Detailed statistical summary per channel
#   - High-quality interactive plots (HTML) & static previews (PNG)
#   - Structured output: dedicated folders per source file
#   - Full logging & fault tolerance
# """

# import os
# import re
# import json
# import logging
# import argparse
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Any, Optional, Union, Tuple  # <--- Fixed: Added Tuple

# import numpy as np
# import pandas as pd
# from scipy.io import loadmat
# import plotly.graph_objects as go
# import plotly.io as pio
# from tqdm import tqdm

# # -------------------------
# # 1. Configuration & Setup
# # -------------------------
# # Default path (can be overridden via CLI)
# DEFAULT_DATA_PATH = r"d:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP"
# OUTPUT_SUBDIR = "analysis_structured_v2"

# # Logging Setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-7s | %(message)s",
#     datefmt='%H:%M:%S'
# )
# logger = logging.getLogger("Inspector")

# # Plotly Template
# pio.templates.default = "plotly_white"
# COLOR_PALETTE = {
#     "DE": "#2980b9",  # Blue
#     "FE": "#e67e22",  # Orange
#     "BA": "#27ae60",  # Green
#     "Unknown": "#7f8c8d" # Grey
# }

# # -------------------------
# # 2. Helper Functions
# # -------------------------

# def safe_filename(name: str) -> str:
#     """Sanitize string for filesystem usage."""
#     return re.sub(r"[^0-9A-Za-z._-]", "_", name)

# def flatten_if_needed(a: Any) -> Optional[np.ndarray]:
#     """Ensure array is 1D numpy array."""
#     if a is None:
#         return None
#     try:
#         arr = np.asarray(a)
#         if arr.size == 0:
#             return None
#         # Check specifically for numeric types
#         if not np.issubdtype(arr.dtype, np.number):
#              return None
#         return arr.ravel()
#     except Exception:
#         return None

# def get_file_stats(filepath: Path) -> Dict[str, Any]:
#     """Get basic OS file stats."""
#     stat = filepath.stat()
#     return {
#         "size_bytes": stat.st_size,
#         "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
#     }

# def identify_channels(mat: Dict[str, Any]) -> Dict[str, Tuple[str, np.ndarray]]:
#     """
#     Heuristically identify Drive End (DE), Fan End (FE), and Base (BA) signals.
#     Returns: { 'DE': (original_key, data_array), ... }
#     """
#     channel_dict = {}
#     # Regex patterns for robustness
#     patterns = {
#         'DE': re.compile(r'(de_time|drive_end|de)', re.IGNORECASE),
#         'FE': re.compile(r'(fe_time|fan_end|fe)', re.IGNORECASE),
#         'BA': re.compile(r'(ba_time|base|ba)', re.IGNORECASE)
#     }
    
#     # First pass: strict regex match
#     for key, val in mat.items():
#         if key.startswith("__") or key.startswith("RPM"): continue # Skip meta & RPM for now
        
#         arr = flatten_if_needed(val)
#         if arr is None or arr.size < 100: continue # Skip tiny/empty arrays
        
#         # Check key against patterns
#         for ch_name, pat in patterns.items():
#             if pat.search(key):
#                 if ch_name not in channel_dict:
#                     channel_dict[ch_name] = (key, arr)
#                 break 
        
#     return channel_dict

# # -------------------------
# # 3. Visualization Logic
# # -------------------------

# def plot_signal(signal: np.ndarray, channel_name: str, file_name: str, sample_rate: int = 12000) -> go.Figure:
#     """Create a professional time-domain plot of the signal."""
#     # Downsample for plotting performance if needed (visual only)
#     max_points = 5000
#     step = max(1, len(signal) // max_points)
    
#     # Time vector
#     t = np.arange(0, len(signal)) / sample_rate
    
#     # Subsample for display
#     sig_vis = signal[::step]
#     t_vis = t[::step]
    
#     # Determine color based on channel type (DE, FE, etc.)
#     # Use first 2 chars to match keys in palette
#     prefix = channel_name[:2].upper()
#     color = COLOR_PALETTE.get(prefix, COLOR_PALETTE["Unknown"])
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=t_vis, y=sig_vis,
#         mode='lines',
#         name=channel_name,
#         line=dict(color=color, width=1.5)
#     ))
    
#     stats_text = (
#         f"<b>Mean:</b> {np.mean(signal):.4f}<br>"
#         f"<b>Std:</b> {np.std(signal):.4f}<br>"
#         f"<b>Peak:</b> {np.max(np.abs(signal)):.4f}<br>"
#         f"<b>RMS:</b> {np.sqrt(np.mean(signal**2)):.4f}"
#     )
    
#     fig.update_layout(
#         title=dict(
#             text=f"<b>{file_name}</b> - {channel_name} Signal",
#             font=dict(size=18)
#         ),
#         xaxis_title="Time (seconds)",
#         yaxis_title="Amplitude (m/s²)",
#         margin=dict(l=60, r=40, t=80, b=60),
#         height=500,
#         width=900,
#         template="plotly_white",
#         annotations=[dict(
#             x=1, y=1, xref="paper", yref="paper",
#             text=stats_text,
#             showarrow=False,
#             align="left",
#             bgcolor="rgba(255, 255, 255, 0.8)",
#             bordercolor="#ddd", borderwidth=1
#         )]
#     )
#     return fig

# def save_plot(fig: go.Figure, output_path: Path):
#     """Save as HTML (interactive) and PNG (static preview)."""
#     # HTML
#     fig.write_html(str(output_path.with_suffix(".html")), include_plotlyjs="cdn", config={'displayModeBar': True})
    
#     # PNG (Robust check for kaleido)
#     try:
#         fig.write_image(str(output_path.with_suffix(".png")), scale=2)
#     except Exception as e:
#         # Fallback: User might not have kaleido installed
#         # logger.debug(f"PNG export skipped (kaleido missing or error): {e}")
#         pass

# # -------------------------
# # 4. Main Execution Logic
# # -------------------------

# def process_files(data_path: Path, output_root: Path):
#     """Main processing loop."""
#     files = sorted(list(data_path.glob("*.mat")))
#     if not files:
#         logger.error(f"No .mat files found in {data_path}")
#         return

#     summary_data = []
#     logger.info(f"Found {len(files)} files. Starting inspection...")
    
#     # Setup global output directory
#     output_root.mkdir(parents=True, exist_ok=True)
    
#     for idx, fpath in enumerate(tqdm(files, desc="Inspecting Files"), 1):
#         file_record = {
#             "id": idx,
#             "filename": fpath.name,
#             "path": str(fpath),
#             "status": "processed",
#             "channels": []
#         }
        
#         try:
#             # 1. Load MAT
#             try:
#                 mat_data = loadmat(str(fpath))
#             except Exception as e:
#                 logger.warning(f"Failed to load {fpath.name}: {e}")
#                 file_record["status"] = "load_error"
#                 file_record["error"] = str(e)
#                 summary_data.append(file_record)
#                 continue
            
#             # 2. Identify Channels
#             channels = identify_channels(mat_data)
#             if not channels:
#                 logger.warning(f"No recognizable signal channels in {fpath.name}")
#                 file_record["status"] = "no_channels"
#                 summary_data.append(file_record)
#                 continue
            
#             # 3. Create File-Specific Output Folder
#             # Folder name: Index_Filename (e.g., 01_97.mat)
#             folder_name = f"{idx:02d}_{safe_filename(fpath.stem)}"
#             file_out_dir = output_root / folder_name
#             file_out_dir.mkdir(parents=True, exist_ok=True)
            
#             # 4. Process Each Channel
#             for ch_label, (mat_key, signal) in channels.items():
#                 # Stats
#                 sig_len = len(signal)
#                 sig_min = float(np.min(signal))
#                 sig_max = float(np.max(signal))
#                 sig_rms = float(np.sqrt(np.mean(signal**2)))
                
#                 ch_info = {
#                     "label": ch_label,
#                     "mat_key": mat_key,
#                     "length": sig_len,
#                     "min": sig_min,
#                     "max": sig_max,
#                     "rms": sig_rms
#                 }
#                 file_record["channels"].append(ch_info)
                
#                 # Plotting
#                 fig = plot_signal(signal, ch_label, fpath.name)
#                 plot_filename = f"{ch_label}_{mat_key}"
#                 save_plot(fig, file_out_dir / plot_filename)
            
#             # Save per-file JSON metadata
#             with open(file_out_dir / "metadata.json", "w", encoding="utf-8") as f:
#                 json.dump(file_record, f, indent=2)
                
#         except Exception as e:
#             logger.exception(f"Unexpected error processing {fpath.name}: {e}")
#             file_record["status"] = "fatal_error"
#             file_record["error"] = str(e)
        
#         summary_data.append(file_record)

#     # 5. Final Global Summary
#     df_summary = pd.json_normalize(
#         summary_data, 
#         record_path=['channels'], 
#         meta=['filename', 'status'], 
#         errors='ignore'
#     )
    
#     # Save Summary Reports
#     csv_path = output_root / "dataset_summary.csv"
#     json_path = output_root / "dataset_summary.json"
    
#     try:
#         df_summary.to_csv(csv_path, index=False)
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(summary_data, f, indent=2)
            
#         logger.info(f"Inspection complete.")
#         logger.info(f"Summary CSV: {csv_path}")
#         logger.info(f"Detailed JSON: {json_path}")
#     except Exception as e:
#         logger.error(f"Failed to save summary reports: {e}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="CWRU Dataset Professional Inspector")
#     parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_PATH, help="Path to folder containing .mat files")
#     parser.add_argument("--output-name", type=str, default=OUTPUT_SUBDIR, help="Name of output folder")
    
#     args = parser.parse_args()
    
#     base_path = Path(args.data_dir)
#     if not base_path.exists():
#         logger.error(f"Data directory not found: {base_path}")
#     else:
#         out_path = base_path / args.output_name
#         process_files(base_path, out_path)











# -*- coding: utf-8 -*-
"""
inspect_cwru_files_final.py
Author: Ali Naderi - 2025 (Final Polish)
Purpose: Professional Inspection of CWRU Dataset Files
Features:
  - Journal-grade plotting aesthetics (matching advanced_report)
  - Overlap-free layout with external legends/stats
  - Robust channel identification
  - Structured output organization
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm

# -------------------------
# 1. Configuration & Styling
# -------------------------
DEFAULT_DATA_PATH = r"d:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP"
OUTPUT_SUBDIR = "analysis_structured_final"

# "Midnight Scientific" Palette (Consistent with advanced_report)
PRO_PALETTE = [
    "#2C3E50",  # Midnight Blue
    "#E74C3C",  # Brick Red
    "#F1C40F",  # Muted Gold
    "#27AE60",  # Jungle Green
    "#8E44AD",  # Royal Purple
]

# Channel-specific color mapping
CHANNEL_COLORS = {
    "DE": "#2980b9",  # Blue
    "FE": "#e67e22",  # Orange
    "BA": "#27ae60",  # Green
    "Unknown": "#7f8c8d"
}

pio.templates.default = "plotly_white"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Inspector")

# -------------------------
# 2. Helpers
# -------------------------

def safe_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", name)

def flatten_if_needed(a: Any) -> Optional[np.ndarray]:
    if a is None: return None
    try:
        arr = np.asarray(a)
        if arr.size == 0: return None
        if not np.issubdtype(arr.dtype, np.number): return None
        return arr.ravel()
    except: return None

def identify_channels(mat: Dict[str, Any]) -> Dict[str, Tuple[str, np.ndarray]]:
    channel_dict = {}
    patterns = {
        'DE': re.compile(r'(de_time|drive_end|de)', re.IGNORECASE),
        'FE': re.compile(r'(fe_time|fan_end|fe)', re.IGNORECASE),
        'BA': re.compile(r'(ba_time|base|ba)', re.IGNORECASE)
    }
    
    for key, val in mat.items():
        if key.startswith("__") or key.startswith("RPM"): continue
        arr = flatten_if_needed(val)
        if arr is None or arr.size < 100: continue
        
        for ch_name, pat in patterns.items():
            if pat.search(key):
                if ch_name not in channel_dict:
                    channel_dict[ch_name] = (key, arr)
                break 
    return channel_dict

# -------------------------
# 3. Visualization Logic
# -------------------------

def plot_signal_professional(signal: np.ndarray, channel_name: str, file_name: str, sample_rate: int = 12000) -> go.Figure:
    """
    Generates a publication-ready time domain plot.
    - Downsamples for performance but keeps visual fidelity.
    - Places stats in an external annotation to avoid overlap.
    """
    # Smart downsampling
    n_samples = len(signal)
    target_points = 8000
    step = max(1, n_samples // target_points)
    
    t = np.arange(0, n_samples) / sample_rate
    
    sig_vis = signal[::step]
    t_vis = t[::step]
    
    # Select color
    prefix = channel_name[:2].upper()
    color = CHANNEL_COLORS.get(prefix, PRO_PALETTE[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_vis, y=sig_vis,
        mode='lines',
        name=channel_name,
        line=dict(color=color, width=1.2),
        showlegend=False 
    ))
    
    # Calculate Stats
    mu = np.mean(signal)
    sigma = np.std(signal)
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    
    # Format Stats Text (HTML)
    stats_html = (
        f"<b>Statistics</b><br>"
        f"Mean: {mu:.4f}<br>"
        f"Std Dev: {sigma:.4f}<br>"
        f"Peak: {peak:.4f}<br>"
        f"RMS: {rms:.4f}"
    )

    # Layout Configuration
    fig.update_layout(
        title=dict(
            text=f"<b>Signal Analysis: {file_name}</b><br><span style='font-size:14px;color:#555'>Channel: {channel_name}</span>",
            font=dict(size=20, family="Arial"),
            y=0.95, x=0.05, xanchor='left'
        ),
        xaxis=dict(
            title="Time (seconds)",
            showgrid=True, gridcolor='#F0F0F0',
            linecolor='#333', linewidth=1.5,
            rangeslider=dict(visible=False) # Can enable for HTML interaction
        ),
        yaxis=dict(
            title="Amplitude (m/s²)",
            showgrid=True, gridcolor='#F0F0F0',
            linecolor='#333', linewidth=1.5,
            zeroline=True, zerolinecolor='#ddd'
        ),
        margin=dict(l=80, r=150, t=80, b=60), # Extra right margin for stats
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        width=1000,
        font=dict(family="Arial, Roboto", size=14, color="#2c3e50")
    )
    
    # Add Stats Annotation (Outside Plot Area)
    fig.add_annotation(
        text=stats_html,
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=1.02, y=1.0, # Positioned to the right
        yanchor='top',
        bgcolor="#f8f9fa",
        bordercolor="#e1e4e8",
        borderwidth=1,
        font=dict(size=12)
    )
    
    return fig

def save_plot_robust(fig: go.Figure, output_path: Path):
    """Saves HTML and attempts PNG export."""
    # Save HTML
    html_path = output_path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn", config={'displayModeBar': True})
    
    # Save PNG (Static)
    png_path = output_path.with_suffix(".png")
    try:
        # Use scale=2 for high-res (approx 300 DPI for screen width)
        fig.write_image(str(png_path), scale=2, width=1000, height=500)
    except Exception:
        pass

# -------------------------
# 4. Processing Loop
# -------------------------
def process_files(data_path: Path, output_root: Path):
    files = sorted(list(data_path.glob("*.mat")))
    if not files:
        logger.error(f"No .mat files found in {data_path}")
        return

    summary_data = []
    logger.info(f"Found {len(files)} files. Output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    
    for idx, fpath in enumerate(tqdm(files, desc="Analyzing"), 1):
        file_record = {
            "id": idx,
            "filename": fpath.name,
            "channels": []
        }
        
        try:
            mat_data = loadmat(str(fpath))
            channels = identify_channels(mat_data)
            
            if not channels:
                logger.warning(f"Skipping {fpath.name}: No valid channels.")
                continue
            
            # Create subfolder
            folder_name = f"{idx:02d}_{safe_filename(fpath.stem)}"
            file_out_dir = output_root / folder_name
            file_out_dir.mkdir(parents=True, exist_ok=True)
            
            for ch_label, (mat_key, signal) in channels.items():
                # Collect stats
                ch_info = {
                    "label": ch_label,
                    "key": mat_key,
                    "samples": len(signal),
                    "rms": float(np.sqrt(np.mean(signal**2)))
                }
                file_record["channels"].append(ch_info)
                
                # Plot
                fig = plot_signal_professional(signal, ch_label, fpath.name)
                plot_name = f"{ch_label}_{safe_filename(mat_key)}"
                save_plot_robust(fig, file_out_dir / plot_name)
            
            # Save file metadata
            with open(file_out_dir / "meta.json", "w") as f:
                json.dump(file_record, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing {fpath.name}: {e}")
            
        summary_data.append(file_record)

    # Global Summary
    df = pd.json_normalize(summary_data, record_path=['channels'], meta=['filename'])
    df.to_csv(output_root / "summary_report.csv", index=False)
    logger.info(f"Completed. Summary saved to {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-name", type=str, default=OUTPUT_SUBDIR)
    args = parser.parse_args()
    
    process_files(Path(args.data_dir), Path(args.data_dir) / args.output_name)