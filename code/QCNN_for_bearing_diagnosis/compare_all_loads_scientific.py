# -*- coding: utf-8 -*-
"""
compare_all_loads_scientific.py
Author: Ali Naderi - 2025
Description: 
    Aggregates results from different working loads (0HP, 1HP, 2HP, 3HP)
    and generates publication-quality comparative figures (IEEE Standard).
    Fixes: Plotly API compatibility & Windows encoding issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
BASE_DIR = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis"

RESULT_PATHS = {
    "0HP": os.path.join(BASE_DIR, r"checkpoints_final_vis_0hp\final_results.csv"),
    "1HP": os.path.join(BASE_DIR, r"checkpoints_final_vis_1hp\final_results.csv"),
    "2HP": os.path.join(BASE_DIR, r"checkpoints_final_vis_2hp\final_results.csv"),
    "3HP": os.path.join(BASE_DIR, r"checkpoints_final_vis_3hp\final_results.csv")
}

OUTPUT_DIR = r"./comparative_results"

class ScientificPlotter:
    """ Enhanced Plotter with IEEE Style """
    PALETTE = px.colors.qualitative.G10
    
    @staticmethod
    def _apply_style(fig, title="", x_title="", y_title="", width=1000, height=700):
        fig.update_layout(
            template="simple_white",
            title=dict(
                text=f"<b>{title}</b>", 
                y=0.96, x=0.5, xanchor='center', yanchor='top',
                font=dict(family="Times New Roman", size=24, color="black")
            ),
            xaxis=dict(
                title=dict(text=f"<b>{x_title}</b>", font=dict(family="Times New Roman", size=20)),
                showline=True, linewidth=2, linecolor='black', mirror=True,
                showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
                tickfont=dict(family="Times New Roman", size=16, color='black')
            ),
            yaxis=dict(
                title=dict(text=f"<b>{y_title}</b>", font=dict(family="Times New Roman", size=20)),
                showline=True, linewidth=2, linecolor='black', mirror=True,
                showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
                tickfont=dict(family="Times New Roman", size=16, color='black')
            ),
            font=dict(family="Times New Roman", color="black"),
            legend=dict(
                font=dict(family="Times New Roman", size=16),
                bordercolor="Black", borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                x=0.02, y=0.02, xanchor='left', yanchor='bottom'
            ),
            margin=dict(l=80, r=40, t=80, b=40),
            width=width, height=height
        )
        return fig

    @staticmethod
    def save(fig, name):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.write_html(os.path.join(OUTPUT_DIR, f"{name}.html"))
        try:
            fig.write_image(os.path.join(OUTPUT_DIR, f"{name}.png"), scale=3)
        except: pass

def load_data():
    df_list = []
    print(f"Checking for result files in {BASE_DIR}...")
    for load_name, path in RESULT_PATHS.items():
        if os.path.exists(path):
            print(f" [OK] Found {load_name}")
            df = pd.read_csv(path)
            if 'acc' in df.columns: df.rename(columns={'acc': 'Accuracy'}, inplace=True)
            if 'snr' in df.columns: df.rename(columns={'snr': 'SNR'}, inplace=True)
            df['Load'] = load_name
            df_list.append(df)
        else:
            print(f" [MISSING] File not found for {load_name}: {path}")
    
    if not df_list:
        return None
    return pd.concat(df_list, ignore_index=True)

def plot_robustness_comparison(df):
    df_num = df[df['SNR'] != 'Clean'].copy()
    df_num['SNR'] = df_num['SNR'].astype(int)
    df_num = df_num.sort_values('SNR')
    
    fig = go.Figure()
    markers = ['circle', 'square', 'diamond', 'triangle-up']
    
    for i, load in enumerate(sorted(df['Load'].unique())):
        data = df_num[df_num['Load'] == load]
        fig.add_trace(go.Scatter(
            x=data['SNR'], y=data['Accuracy'],
            mode='lines+markers', name=load,
            line=dict(width=4, color=ScientificPlotter.PALETTE[i]),
            marker=dict(size=12, symbol=markers[i % len(markers)], line=dict(width=2, color='white'))
        ))

    fig = ScientificPlotter._apply_style(fig, "Noise Robustness Across Working Loads", 
                                         "Signal-to-Noise Ratio (dB)", "Accuracy (%)")
    fig.update_yaxes(range=[90, 101]) 
    fig.update_xaxes(tickmode='linear', dtick=2)
    ScientificPlotter.save(fig, "Comparison_Robustness_Curve")

def plot_critical_snr_bars(df):
    target_snrs = ['-10', '-6', '0', 'Clean']
    df['SNR_Str'] = df['SNR'].astype(str)
    data = df[df['SNR_Str'].isin(target_snrs)].copy()
    
    data['SNR_Str'] = pd.Categorical(data['SNR_Str'], categories=target_snrs, ordered=True)
    data = data.sort_values(['SNR_Str', 'Load'])

    fig = px.bar(data, x="SNR_Str", y="Accuracy", color="Load", barmode="group",
                 color_discrete_sequence=ScientificPlotter.PALETTE,
                 text_auto='.1f')
    
    fig = ScientificPlotter._apply_style(fig, "Performance at Critical Noise Levels", 
                                         "Noise Level (dB)", "Accuracy (%)")
    fig.update_traces(textposition='outside', textfont=dict(family="Times New Roman", size=14, color="black"), width=0.2)
    fig.update_yaxes(range=[85, 105])
    ScientificPlotter.save(fig, "Comparison_Bar_Chart")

def plot_3d_landscape(df):
    df_num = df[df['SNR'] != 'Clean'].copy()
    df_num['SNR'] = df_num['SNR'].astype(int)
    
    load_map = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}
    df_num['Load_Int'] = df_num['Load'].map(load_map)
    
    x = np.sort(df_num['SNR'].unique())
    y = np.sort(df_num['Load_Int'].unique())
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i, snr in enumerate(x):
        for j, load in enumerate(y):
            val = df_num[(df_num['SNR'] == snr) & (df_num['Load_Int'] == load)]['Accuracy'].values
            if len(val) > 0:
                Z[j, i] = val[0]
            else:
                Z[j, i] = np.nan

    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale='Plasma', opacity=0.95,
        contours={"z": {"show": True, "start": 90, "end": 100, "size": 1, "color":"white"}}
    )])

    # FIX: Corrected API for 3D axis titles (removed deprecated 'titlefont')
    fig.update_layout(
        title=dict(text="<b>3D Performance Landscape</b>", y=0.9, x=0.5, font=dict(family="Times New Roman", size=24, color="black")),
        scene=dict(
            xaxis=dict(title=dict(text='<b>SNR (dB)</b>', font=dict(size=16)), tickmode='linear', dtick=2, backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title=dict(text='<b>Load (HP)</b>', font=dict(size=16)), tickvals=[0,1,2,3], ticktext=['0HP','1HP','2HP','3HP'], backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(title=dict(text='<b>Accuracy (%)</b>', font=dict(size=16)), range=[90, 100], backgroundcolor="white", gridcolor="lightgray"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        font=dict(family="Times New Roman"),
        width=1000, height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    ScientificPlotter.save(fig, "Comparison_3D_Surface")

def main():
    print("=== Generating Comparative Analysis Charts (Scientific Edition) ===")
    
    try:
        df = load_data()
        if df is None: return
        
        print("\nGenerating Plots...")
        plot_robustness_comparison(df)
        print(" -> Generated Robustness Curve")
        
        plot_critical_snr_bars(df)
        print(" -> Generated Critical Bar Chart")
        
        plot_3d_landscape(df)
        print(" -> Generated 3D Landscape")
        
        summary_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
        df.to_csv(summary_path, index=False)
        print(f" -> Saved summary data to {summary_path}")
        
        print(f"\n[DONE] All comparative figures saved to: {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()