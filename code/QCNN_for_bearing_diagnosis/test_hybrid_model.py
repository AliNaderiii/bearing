# -*- coding: utf-8 -*-
"""
evaluate_scientific_final.py
Author: Ali Naderi - 2025 (Final Q1 Journal Edition)
Description: 
    Generates high-end visualization artifacts for Bearing Fault Diagnosis.
    Includes:
    1. Confusion Matrix (Heatmap)
    2. Class Sensitivity Radar Chart
    3. Multiclass ROC Curves
    4. T-SNE Feature Clustering
    5. Robustness Curve (Accuracy vs SNR)
    6. **NEW** Chord Diagram (Misclassification Flow) - Highly recommended for Q1 papers.
    
    Style: IEEE Standard (Times New Roman, High Contrast).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Chord Diagram Libraries
try:
    import holoviews as hv
    from holoviews import opts, dim
    hv.extension('bokeh')
    CHORD_AVAILABLE = True
except ImportError:
    print("[Warning] 'holoviews' or 'bokeh' not installed. Chord Diagram will be skipped.")
    print("Install via: pip install holoviews bokeh")
    CHORD_AVAILABLE = False

# --- 1. SETUP PATHS & IMPORTS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
if os.path.exists(os.path.join(base_dir, 'utils')):
    sys.path.append(os.path.join(base_dir, 'utils'))
if os.path.exists(os.path.join(base_dir, 'Model')):
    sys.path.append(os.path.join(base_dir, 'Model'))

try:
    try:
        from utils.DatasetLoader import RobustBearingDataset, collate_fn
    except ImportError:
        from utils.DatasetLoader import RobustBearingDataset, collate_fn

    try:
        from Model.model import QDCA_Hybrid_Pro
    except ImportError:
        from Model.model import QDCA_Hybrid_Pro

except ImportError as e:
    print(f"[CRITICAL ERROR] Import failed: {e}")
    sys.exit(1)

# --- 2. CONFIGURATION ---
# Adjust paths if necessary
DEFAULT_NPZ = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
DEFAULT_META = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\meta.json"
# Using the best seed model from the Ultra run
CHECKPOINT_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_3hp\model_seed42.pth"
OUTPUT_DIR = r"./paper_results_final_3hp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. SCIENTIFIC PLOTTER (PLOTLY) ---
class ScientificPlotter:
    """ Generates IEEE-style Plotly figures. """
    PALETTE = px.colors.qualitative.G10
    
    @staticmethod
    def _apply_ieee_style(fig, title="", x_title="", y_title="", width=900, height=750):
        fig.update_layout(
            template="simple_white", 
            title=dict(
                text=f"<b>{title}</b>", 
                y=0.96, x=0.5, xanchor='center', yanchor='top',
                font=dict(family="Times New Roman", size=22, color="black")
            ),
            xaxis=dict(
                title=dict(text=f"<b>{x_title}</b>", font=dict(family="Times New Roman", size=18)),
                showline=True, linewidth=2, linecolor='black', mirror=True,
                showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
                tickfont=dict(family="Times New Roman", size=16, color='black')
            ),
            yaxis=dict(
                title=dict(text=f"<b>{y_title}</b>", font=dict(family="Times New Roman", size=18)),
                showline=True, linewidth=2, linecolor='black', mirror=True,
                showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
                tickfont=dict(family="Times New Roman", size=16, color='black')
            ),
            font=dict(family="Times New Roman", color="black"),
            legend=dict(
                font=dict(family="Times New Roman", size=14),
                bordercolor="Black", borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                x=0.99, y=0.01, xanchor='right', yanchor='bottom'
            ),
            margin=dict(l=80, r=40, t=80, b=80),
            width=width, height=height
        )
        return fig

    @staticmethod
    def save(fig, name, directory):
        os.makedirs(directory, exist_ok=True)
        fig.write_html(os.path.join(directory, f"{name}.html"))
        try:
            fig.write_image(os.path.join(directory, f"{name}.png"), scale=3)
        except: pass

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, snr, save_dir):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Smart Annotations
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i][j]
                text_color = "white" if val > 50 else "black"
                txt = f"{val:.1f}" if val > 0.0 else "0"
                annotations.append(dict(
                    x=class_names[j], y=class_names[i], text=txt,
                    font=dict(family="Times New Roman", size=14, color=text_color),
                    showarrow=False
                ))

        fig = go.Figure(data=go.Heatmap(
            z=cm_norm, x=class_names, y=class_names,
            colorscale='Blues', showscale=True,
            colorbar=dict(
                title=dict(text='<b>Acc (%)</b>', side='right', font=dict(family="Times New Roman", size=16)),
                tickfont=dict(family="Times New Roman", size=14)
            )
        ))
        
        fig.update_layout(annotations=annotations)
        fig = ScientificPlotter._apply_ieee_style(fig, title=f"Confusion Matrix (SNR = {snr} dB)",
                                                  x_title="Predicted Class", y_title="True Class")
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"))
        ScientificPlotter.save(fig, f"CM_{snr}dB", save_dir)

    @staticmethod
    def plot_radar_chart(y_true, y_pred, class_names, snr, save_dir):
        cm = confusion_matrix(y_true, y_pred)
        recall = cm.diagonal() / cm.sum(axis=1) * 100
        
        categories = class_names + [class_names[0]]
        values = list(recall) + [recall[0]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories,
            fill='toself', name='Recall',
            line=dict(color='#1f77b4', width=3),
            fillcolor='rgba(31, 119, 180, 0.4)'
        ))
        
        fig.update_layout(
            template="simple_white",
            title=dict(text=f"<b>Class Sensitivity Radar (SNR = {snr} dB)</b>", y=0.96, x=0.5, xanchor='center', font=dict(family="Times New Roman", size=20)),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='lightgray', tickfont=dict(family="Times New Roman", size=12)),
                angularaxis=dict(tickfont=dict(family="Times New Roman", size=14, weight="bold"))
            ),
            font=dict(family="Times New Roman"),
            width=800, height=800
        )
        ScientificPlotter.save(fig, f"Radar_{snr}dB", save_dir)

    @staticmethod
    def plot_robustness_curve(results, save_dir):
        df = pd.DataFrame(results)
        df = df[df['SNR'] != 'Clean'].copy()
        df['SNR'] = df['SNR'].astype(int)
        df = df.sort_values('SNR')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['SNR'], y=df['Accuracy'],
            mode='lines+markers', name='QDCA-Net',
            line=dict(color='#D62728', width=4),
            marker=dict(size=14, symbol='diamond', line=dict(width=2, color='white'))
        ))
        
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['SNR'], y=row['Accuracy'],
                text=f"<b>{row['Accuracy']:.1f}%</b>",
                showarrow=False, yshift=20,
                font=dict(family="Times New Roman", size=14, color="black")
            )

        fig = ScientificPlotter._apply_ieee_style(fig, title="Noise Robustness Analysis", 
                                                  x_title="Signal-to-Noise Ratio (dB)", y_title="Accuracy (%)",
                                                  width=1000, height=600)
        fig.update_yaxes(range=[min(df['Accuracy'])-5, 105])
        fig.update_xaxes(tickmode='linear', dtick=2)
        ScientificPlotter.save(fig, "Robustness_Curve", save_dir)

    @staticmethod
    def plot_tsne(features, labels, class_names, snr, save_dir):
        print(f"   > Computing T-SNE for {snr}dB...")
        if len(features) > 2000:
            idx = np.random.choice(len(features), 2000, replace=False)
            features = features[idx]
            labels = labels[idx]
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(features)
        
        label_strs = [class_names[i] for i in labels]
        df = pd.DataFrame({'D1': emb[:, 0], 'D2': emb[:, 1], 'Fault Type': label_strs})
        
        fig = px.scatter(df, x='D1', y='D2', color='Fault Type', symbol='Fault Type',
                         color_discrete_sequence=ScientificPlotter.PALETTE)
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white'), opacity=0.9))
        
        fig = ScientificPlotter._apply_ieee_style(fig, title=f"Feature Clusters (SNR = {snr} dB)", 
                                                  x_title="Dimension 1", y_title="Dimension 2")
        fig.update_layout(legend_title_text='')
        ScientificPlotter.save(fig, f"TSNE_{snr}dB", save_dir)

    @staticmethod
    def plot_multiclass_roc(y_true, y_probs, class_names, snr, save_dir):
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        colors = px.colors.qualitative.Prism 
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc_score = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{class_names[i]} (AUC={auc_score:.3f})",
                mode='lines', line=dict(width=2, color=colors[i % len(colors)])
            ))
            
        fig = ScientificPlotter._apply_ieee_style(fig, title=f"Multiclass ROC Curves (SNR = {snr} dB)",
                                                  x_title="False Positive Rate", y_title="True Positive Rate")
        ScientificPlotter.save(fig, f"ROC_{snr}dB", save_dir)

# --- 4. CHORD VISUALIZER (HoloViews) ---
class ChordVisualizer:
    """ Generates Flow Diagrams for Misclassification Analysis """
    
    @staticmethod
    def plot_chord(y_true, y_pred, class_names, snr, save_dir):
        if not CHORD_AVAILABLE: return
        
        print(f"   > Generating Chord Diagram for {snr}dB...")
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert CM to Source-Target-Value format
        links = []
        for i, source in enumerate(class_names):
            for j, target in enumerate(class_names):
                count = cm[i, j]
                # Filter out very small flows to keep chart clean (optional)
                if count > 0:
                    links.append((source, target, int(count)))
        
        df_links = pd.DataFrame(links, columns=['source', 'target', 'value'])
        
        # Create Chord
        chord = hv.Chord(df_links)
        
        # Style
        chord.opts(
            opts.Chord(
                cmap='Category20', 
                edge_cmap='Category20', 
                edge_color=dim('source').str(), 
                labels='source', 
                node_color=dim('index').str(), 
                title=f'Fault Misclassification Flow (SNR = {snr} dB)',
                width=800, height=800,
                fontsize={'labels': 12, 'title': 16, 'xticks': 12, 'yticks': 12}
            )
        )
        
        out_path = os.path.join(save_dir, f"Chord_{snr}dB.html")
        hv.save(chord, out_path)
        # Note: PNG saving for HoloViews requires selenium/phantomjs, HTML is safer default.

# --- 5. MAIN LOGIC ---
def load_assets():
    if not os.path.exists(DEFAULT_META):
        print("[Error] Meta file missing.")
        sys.exit(1)
    
    with open(DEFAULT_META, 'r') as f: meta = json.load(f)
    class_names = [k for k, v in sorted(meta['class_map'].items(), key=lambda item: item[1])]
    
    # Check feat_dim
    try:
        temp_ds = RobustBearingDataset(DEFAULT_NPZ, mode='train', load_into_memory=False)
        feat_dim = temp_ds[0][1].shape[0]
    except: feat_dim = 33

    if not os.path.exists(CHECKPOINT_PATH):
        print("[Error] Checkpoint missing.")
        sys.exit(1)

    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=len(class_names), feat_dim=feat_dim).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model, class_names

def evaluate(model, snr):
    try:
        ds = RobustBearingDataset(DEFAULT_NPZ, mode='test', snr=snr, load_into_memory=True, instance_norm=True)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    except: return None

    preds, labels, probs = [], [], []
    features_list = []
    
    def hook(m, i, o): features_list.append(o.detach().cpu().numpy())
    handle = model.classifier[0].register_forward_hook(hook) # Hook LayerNorm
    
    with torch.no_grad():
        for x, f, y in tqdm(loader, desc=f"   Eval SNR {snr}", leave=False):
            x, f = x.to(DEVICE), f.to(DEVICE)
            logits = model(x, f)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(prob.cpu().numpy())
            
    handle.remove()
    return {
        'y_true': np.array(labels), 'y_pred': np.array(preds),
        'y_probs': np.array(probs), 'features': np.concatenate(features_list, axis=0),
        'acc': accuracy_score(labels, preds) * 100
    }

def main():
    print("=== Final Scientific Evaluation Pipeline ===")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    model, class_names = load_assets()
    
    # 1. Detailed Plots (-10dB and 0dB)
    for snr in [-10, 0]:
        print(f"\n[1] Processing Detailed Analysis for SNR = {snr} dB...")
        res = evaluate(model, snr)
        if not res: continue
        
        ScientificPlotter.plot_confusion_matrix(res['y_true'], res['y_pred'], class_names, snr, OUTPUT_DIR)
        ScientificPlotter.plot_radar_chart(res['y_true'], res['y_pred'], class_names, snr, OUTPUT_DIR)
        ScientificPlotter.plot_multiclass_roc(res['y_true'], res['y_probs'], class_names, snr, OUTPUT_DIR)
        ScientificPlotter.plot_tsne(res['features'], res['y_true'], class_names, snr, OUTPUT_DIR)
        
        # ** NEW: Chord Diagram **
        if snr == -10:
            ChordVisualizer.plot_chord(res['y_true'], res['y_pred'], class_names, snr, OUTPUT_DIR)

    # 2. Overall Robustness Curve
    print("\n[2] Generating Full Robustness Curve...")
    snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6]
    curve_data = []
    for snr in tqdm(snrs):
        res = evaluate(model, snr)
        if res: curve_data.append({'SNR': snr, 'Accuracy': res['acc']})
        
    ScientificPlotter.plot_robustness_curve(curve_data, OUTPUT_DIR)
    print(f"\n[DONE] All figures (including Chord Diagram) saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

