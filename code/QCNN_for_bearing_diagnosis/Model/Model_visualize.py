# -*- coding: utf-8 -*-
"""
visual_model_architecture_final.py

Author: Ali Naderi - 2025 (refactored, enhanced)
Purpose:
    Generate publication-ready architecture diagrams (macro & micro views)
    for QDCA_Hybrid_Pro_v3 model used in bearing-diagnosis experiments.

Key enhancements in this version:
 - Increased DPI to 600 for high-quality PNG output.
 - Default format changed to PNG for better raster quality in publications.
 - Optimized layout with increased nodesep (0.8) and ranksep (1.0) for clarity.
 - Added fixedsize, width, height to key nodes to prevent label overflow and ensure consistent sizing.
 - Refined color palette for better print and color-blind accessibility.
 - Improved labels with multi-line (\n) for readability.
 - Added global node/edge defaults for professional consistency.
 - Metadata includes model summary if provided.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import logging
import os
import shutil
import sys
from typing import Dict, Optional, Tuple

try:
    from graphviz import Digraph
except Exception as e:
    raise RuntimeError("graphviz Python package not found. Install via `pip install graphviz`.") from e

LOG = logging.getLogger("visual_arch")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(ch)

# Enhanced color palette (optimized for color-blind accessibility, print, and professionalism)
PALETTE = {
    "input": "#E3F2FD",    # Light blue
    "conv": "#FFE4B5",     # Light orange (softer)
    "res": "#90EE90",      # Light green
    "transformer": "#DDA0DD",  # Light purple
    "fusion": "#FFC0CB",   # Light pink
    "out": "#AFEEEE",      # Light teal
    "edge": "#2F4F4F",     # Dark slate gray
    "gate": "#FFFACD",     # Light yellow
    "white": "#FFFFFF",
    "border": "#000000",   # Black for sharper borders
}
FONT_DEFAULT = "Helvetica"

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def check_graphviz_executable() -> bool:
    return shutil.which("dot") is not None

def safe_import_model(module_path: Optional[str], class_name: Optional[str]):
    if not module_path or not class_name:
        return None, "N/A"
    module_path = os.path.abspath(module_path)
    if not os.path.exists(module_path):
        return None, f"Module file not found: {module_path}"
    spec = importlib.util.spec_from_file_location("visual_model_module", module_path)
    if spec is None or spec.loader is None:
        return None, f"Failed to create spec for {module_path}"
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        return None, f"Module import failed: {e}"
    if not hasattr(mod, class_name):
        return None, f"Class '{class_name}' not found in module."
    cls = getattr(mod, class_name)
    summary = ""
    try:
        # Attempt lightweight instantiation (best-effort)
        inst = cls()
        params = sum(p.numel() for p in inst.parameters()) if hasattr(inst, "parameters") else None
        summary = f"Instantiated {class_name}; params_total={params if params is not None else 'unknown'}"
    except Exception as e:
        summary = f"Model class found but instantiation failed: {e}"
    return cls, summary

def style_node(dot: Digraph, name: str, label: str, *,
               shape: str = "box", fillcolor: str = PALETTE["white"],
               fontcolor: str = "black", fontname: str = FONT_DEFAULT,
               fontsize: Optional[int] = 10, width: Optional[float] = 1.2,
               height: Optional[float] = 0.5, margin: Optional[str] = "0.1,0.05",
               fixedsize: bool = True, **attrs) -> None:
    node_attrs = {
        "shape": shape,
        "style": "filled, rounded",
        "fillcolor": fillcolor,
        "fontname": fontname,
        "fontcolor": fontcolor,
        "color": PALETTE["border"],
        "penwidth": "1.5",  # Thicker borders for professionalism
        "fixedsize": "true" if fixedsize else "false",
    }
    if fontsize is not None:
        node_attrs["fontsize"] = str(fontsize)
    if width is not None:
        node_attrs["width"] = str(width)
    if height is not None:
        node_attrs["height"] = str(height)
    if margin is not None:
        node_attrs["margin"] = str(margin)
    node_attrs.update(attrs)
    dot.node(name, label=label, **node_attrs)

def add_legend(dot: Digraph, title: str = "Legend") -> None:
    with dot.subgraph(name="cluster_legend") as leg:
        leg.attr(label=title, fontsize="12", fontname=FONT_DEFAULT, style="rounded", color=PALETTE["border"])
        style_node(leg, "_L_input", "Input", shape="box", fillcolor=PALETTE["input"], fontsize=9, width=0.8, height=0.4)
        style_node(leg, "_L_conv", "Quadratic Conv / Branches", shape="box", fillcolor=PALETTE["conv"], fontsize=9, width=0.8, height=0.4)
        style_node(leg, "_L_res", "QResBlock / Attention", shape="box", fillcolor=PALETTE["res"], fontsize=9, width=0.8, height=0.4)
        style_node(leg, "_L_transformer", "Transformer", shape="box", fillcolor=PALETTE["transformer"], fontsize=9, width=0.8, height=0.4)
        style_node(leg, "_L_fusion", "Fusion / Gate", shape="box", fillcolor=PALETTE["fusion"], fontsize=9, width=0.8, height=0.4)
        leg.attr(rank="same")

def build_main_architecture(dot: Digraph) -> None:
    dot.attr(rankdir="TB", nodesep="0.8", ranksep="1.0", dpi=str(dot.graph_attr.get('dpi', '600')), compound="true", size="8,10", ratio="fill")
    dot.attr('node', fontsize='10', fontname=FONT_DEFAULT)
    dot.attr('edge', fontsize='9', fontname=FONT_DEFAULT, color=PALETTE["edge"])
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(style="invis")
        style_node(c, "Input", "Raw Signal\n(B, 3, 2048)", shape="note", fillcolor=PALETTE["input"])
        style_node(c, "Feat_Input", "Engineering Features\n(B, 33)", shape="note", fillcolor=PALETTE["input"])
    with dot.subgraph(name="cluster_stem") as c:
        c.attr(label="Stem", fontsize="12", fontname=FONT_DEFAULT, style="dashed", color=PALETTE["border"])
        style_node(c, "Stem_Conv", "QuadraticConv1d\n(k=15, s=2)\n→ (B,32,1024)", fillcolor=PALETTE["conv"])
        style_node(c, "Stem_BN_Act", "BN + MetaAconC", fillcolor=PALETTE["white"])
        style_node(c, "Stem_Pool", "MaxPool1d(s=2)\n→ (B,32,512)", fillcolor=PALETTE["white"])
    dot.edge("Input", "Stem_Conv")
    dot.edge("Stem_Conv", "Stem_BN_Act")
    dot.edge("Stem_BN_Act", "Stem_Pool")
    with dot.subgraph(name="cluster_multiscale") as c:
        c.attr(label="Multi-scale Branches", fontsize="12", fontname=FONT_DEFAULT, style="dashed", color=PALETTE["border"])
        style_node(c, "Branch_L", "Branch Low-freq\nQConv (k=31) + BN + AconC + Pool\n→ (B,64,256)", fillcolor=PALETTE["conv"])
        style_node(c, "Branch_M", "Branch Mid-freq\nQConv (k=15) + BN + AconC + Pool\n→ (B,64,256)", fillcolor=PALETTE["conv"])
        style_node(c, "Branch_H", "Branch High-freq\nQConv (k=7) + BN + AconC + Pool\n→ (B,64,256)", fillcolor=PALETTE["conv"])
    dot.edge("Stem_Pool", "Branch_L")
    dot.edge("Stem_Pool", "Branch_M")
    dot.edge("Stem_Pool", "Branch_H")
    style_node(dot, "Fusion", "Adaptive Fusion\n(Average + Conv1d(64→128, k=1))\n→ (B,128,256)", shape="diamond", fillcolor=PALETTE["fusion"])
    dot.edge("Branch_L", "Fusion")
    dot.edge("Branch_M", "Fusion")
    dot.edge("Branch_H", "Fusion")
    with dot.subgraph(name="cluster_deep") as c:
        c.attr(label="Deep Denoising Stage", fontsize="12", fontname=FONT_DEFAULT, style="dashed", color=PALETTE["border"])
        style_node(c, "Res1", "QResBlock_Pro(128)\n(See Fig. B)\n→ (B,128,256)", fillcolor=PALETTE["res"])
        style_node(c, "Pool", "MaxPool1d(s=2)\n→ (B,128,128)", fillcolor=PALETTE["white"])
        style_node(c, "Res2", "QResBlock_Pro(128)\n→ (B,128,128)", fillcolor=PALETTE["res"])
    dot.edge("Fusion", "Res1")
    dot.edge("Res1", "Pool")
    dot.edge("Pool", "Res2")
    with dot.subgraph(name="cluster_transformer") as c:
        c.attr(label="Transformer", fontsize="12", fontname=FONT_DEFAULT, style="dashed", color=PALETTE["border"])
        style_node(c, "Permute", "Permute (B,128,128)\n+ Pos Embed", fillcolor=PALETTE["white"])
        style_node(c, "Trans_Enc", "TransformerEncoder\n(2 layers, h=4, ff=512, drop=0.2)\n→ (B,128,128)", fillcolor=PALETTE["transformer"])
        style_node(c, "Mean_Pool", "Global Mean Pool (dim=1)\n→ (B,128)", fillcolor=PALETTE["white"])
    dot.edge("Res2", "Permute", label="CNN Out")
    dot.edge("Permute", "Trans_Enc")
    dot.edge("Trans_Enc", "Mean_Pool")
    style_node(dot, "Aux_Net", "Auxiliary MLP\n(Linear(33→64) + BN + SiLU + Drop(0.3)\n+ Linear(64→128) + SiLU)\n→ (B,128)", fillcolor=PALETTE["conv"])
    dot.edge("Feat_Input", "Aux_Net")
    style_node(dot, "Gate", "Gated Fusion\n(See Fig. C)", shape="circle", width=1.0, height=1.0, fillcolor=PALETTE["fusion"])
    dot.edge("Mean_Pool", "Gate", label="Deep Rep\n(B,128)")
    dot.edge("Aux_Net", "Gate", label="Aux Rep\n(B,128)")
    style_node(dot, "Classifier", "Classifier\n(LayerNorm + Dropout(0.3)\n+ Linear(128→10))", fillcolor=PALETTE["out"])
    style_node(dot, "Output", "Logits\n(B, 10)", shape="oval", fillcolor=PALETTE["out"])
    dot.edge("Gate", "Classifier", label="Fused Rep\n(B,128)")
    dot.edge("Classifier", "Output")
    add_legend(dot)

def build_qresblock_detail(dot: Digraph) -> None:
    dot.attr(rankdir="LR", nodesep="0.8", ranksep="1.0", dpi=str(dot.graph_attr.get("dpi", "600")))
    dot.attr('node', fontsize='10', fontname=FONT_DEFAULT)
    dot.attr('edge', fontsize='9', fontname=FONT_DEFAULT, color=PALETTE["edge"])
    style_node(dot, "In", "Input\n(B, C, L)", shape="circle", fillcolor=PALETTE["input"], width=0.8, height=0.8)
    style_node(dot, "QConv1", "QuadraticConv1d\n(k=3, p=1)", fillcolor=PALETTE["conv"])
    style_node(dot, "BN1", "BatchNorm1d", shape="box", fillcolor=PALETTE["white"])
    style_node(dot, "Act1", "MetaAconC\n(Adaptive)", shape="box", fillcolor=PALETTE["white"])
    style_node(dot, "QConv2", "QuadraticConv1d\n(k=3, p=1)", fillcolor=PALETTE["conv"])
    style_node(dot, "BN2", "BatchNorm1d", shape="box", fillcolor=PALETTE["white"])
    style_node(dot, "CoordAtt", "CoordinateAttention\n(Global Context)", fillcolor=PALETTE["res"])
    style_node(dot, "Shrink", "SoftThresholding\n(Learnable Denoising)", fillcolor=PALETTE["res"])
    style_node(dot, "Add", "+", shape="circle", width=0.6, height=0.6, fontsize=14)
    style_node(dot, "Act2", "MetaAconC", shape="doubleoctagon", fillcolor=PALETTE["white"])
    style_node(dot, "Out", "Output\n(B, C, L)", shape="circle", fillcolor=PALETTE["out"], width=0.8, height=0.8)
    dot.edge("In", "QConv1")
    dot.edge("QConv1", "BN1")
    dot.edge("BN1", "Act1")
    dot.edge("Act1", "QConv2")
    dot.edge("QConv2", "BN2")
    dot.edge("BN2", "CoordAtt")
    dot.edge("CoordAtt", "Shrink")
    dot.edge("Shrink", "Add")
    dot.edge("In", "Add", label="Residual", style="dashed")
    dot.edge("Add", "Act2")
    dot.edge("Act2", "Out")

def build_fusion_gate_detail(dot: Digraph) -> None:
    dot.attr(rankdir="TB", nodesep="0.8", ranksep="1.0", dpi=str(dot.graph_attr.get("dpi", "600")))
    dot.attr('node', fontsize='10', fontname=FONT_DEFAULT)
    dot.attr('edge', fontsize='9', fontname=FONT_DEFAULT, color=PALETTE["edge"])
    style_node(dot, "Deep", "Deep Rep\n(B, 128)", shape="note", fillcolor=PALETTE["transformer"])
    style_node(dot, "Aux", "Aux Rep\n(B, 128)", shape="note", fillcolor=PALETTE["conv"])
    style_node(dot, "Concat", "Concatenate\n(B, 256)", shape="invtrapezium", fillcolor=PALETTE["white"])
    with dot.subgraph(name="cluster_gate") as c:
        c.attr(label="Gate Controller", style="dashed, rounded", color=PALETTE["border"])
        style_node(c, "Linear", "Linear(256→1)", shape="box")
        style_node(c, "Sigmoid", "Sigmoid → α (B,1)", shape="box", fillcolor=PALETTE["gate"])
    style_node(dot, "Mul_Deep", "× (1 - α)", shape="circle", width=0.9, height=0.9, fontsize=12)
    style_node(dot, "Mul_Aux", "× α", shape="circle", width=0.9, height=0.9, fontsize=12)
    style_node(dot, "Add", "+", shape="circle", width=0.6, height=0.6, fontsize=14)
    style_node(dot, "Out", "Fused Rep\n(B,128)", shape="oval", fillcolor=PALETTE["out"])
    # Flow
    dot.edge("Deep", "Concat")
    dot.edge("Aux", "Concat")
    dot.edge("Concat", "Linear")
    dot.edge("Linear", "Sigmoid")
    # Gate applications with edge labels
    dot.edge("Sigmoid", "Mul_Deep", label="1 - α")
    dot.edge("Sigmoid", "Mul_Aux", label="α")
    dot.edge("Deep", "Mul_Deep", style="dashed")
    dot.edge("Aux", "Mul_Aux", style="dashed")
    dot.edge("Mul_Deep", "Add")
    dot.edge("Mul_Aux", "Add")
    dot.edge("Add", "Out")

def generate_diagrams(output_dir: str,
                      model_path: Optional[str] = None,
                      model_class: Optional[str] = None,
                      fmt: str = "png",
                      dpi: int = 600,
                      verbose: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    ensure_dir(output_dir)
    base_name = f"architecture_{timestamp()}"
    metadata: Dict[str, str] = {"generated_at": datetime.datetime.now().isoformat()}
    files: Dict[str, str] = {}
    has_dot = check_graphviz_executable()
    if not has_dot:
        LOG.warning("Graphviz 'dot' executable not found in PATH. Rendering may fail. Install Graphviz on your OS.")
    model_summary = "N/A"
    if model_path and model_class:
        cls_or_none, summary = safe_import_model(model_path, model_class)
        model_summary = summary if summary is not None else "N/A"
        metadata["model_summary"] = model_summary
        if cls_or_none is None:
            LOG.warning("Model introspection incomplete: %s", summary)
        else:
            LOG.info("Model introspection: %s", summary)
    else:
        metadata["model_summary"] = "N/A"
    main_dot = Digraph(name="Main_Arch", format=fmt)
    main_dot.attr(dpi=str(dpi))
    try:
        build_main_architecture(main_dot)
        main_path = os.path.join(output_dir, f"{base_name}_main")
        main_dot.render(main_path, cleanup=True, view=False)
        files["main_architecture"] = f"{main_path}.{fmt}"
        LOG.info("Saved main architecture -> %s", files["main_architecture"])
    except Exception as e:
        LOG.error("Failed to render main architecture: %s", e)
        files["main_architecture"] = f"ERROR: {e}"
    qres_dot = Digraph(name="QResBlock", format=fmt)
    qres_dot.attr(dpi=str(dpi))
    try:
        build_qresblock_detail(qres_dot)
        qres_path = os.path.join(output_dir, f"{base_name}_qresblock")
        qres_dot.render(qres_path, cleanup=True, view=False)
        files["qresblock_detail"] = f"{qres_path}.{fmt}"
        LOG.info("Saved QResBlock detail -> %s", files["qresblock_detail"])
    except Exception as e:
        LOG.error("Failed to render QResBlock detail: %s", e)
        files["qresblock_detail"] = f"ERROR: {e}"
    gate_dot = Digraph(name="FusionGate", format=fmt)
    gate_dot.attr(dpi=str(dpi))
    try:
        build_fusion_gate_detail(gate_dot)
        gate_path = os.path.join(output_dir, f"{base_name}_fusion_gate")
        gate_dot.render(gate_path, cleanup=True, view=False)
        files["fusion_gate_detail"] = f"{gate_path}.{fmt}"
        LOG.info("Saved Fusion Gate detail -> %s", files["fusion_gate_detail"])
    except Exception as e:
        LOG.error("Failed to render Fusion Gate detail: %s", e)
        files["fusion_gate_detail"] = f"ERROR: {e}"
    metadata["files"] = files
    meta_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        LOG.info("Saved metadata -> %s", meta_path)
        files["metadata"] = meta_path
    except Exception as e:
        LOG.error("Failed to write metadata: %s", e)
        files["metadata"] = f"ERROR: {e}"
    return files, metadata

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate high-quality, publication-ready architecture diagrams for QDCA_Hybrid_Pro_v3.")
    ap.add_argument("--model-path", type=str, default=None, help="Path to model .py file for introspection (optional).")
    ap.add_argument("--model-class", type=str, default="QDCA_Hybrid_Pro", help="Model class name in the module (optional).")
    ap.add_argument("--output-dir", type=str, default="model_visualizations", help="Directory to save generated figures.")
    ap.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format (png for high-quality raster).")
    ap.add_argument("--dpi", type=int, default=600, help="DPI for raster formats (higher for quality).")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    LOG.info("Starting diagram generation...")
    LOG.debug("Args: %s", vars(args))
    files, meta = generate_diagrams(output_dir=args.output_dir,
                                    model_path=args.model_path,
                                    model_class=args.model_class,
                                    fmt=args.format,
                                    dpi=args.dpi,
                                    verbose=args.verbose)
    LOG.info("Generation finished. Files produced:")
    for k, v in files.items():
        LOG.info("  - %s: %s", k, v)
    LOG.info("Metadata summary:")
    LOG.info("  %s", json.dumps({k: meta.get(k) for k in ["generated_at", "model_summary"]}, indent=2))
