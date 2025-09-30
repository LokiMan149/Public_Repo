# app.py — Streamlit hover PNG viewer (CSV-based) with fixed XY limits & many tabs
# --------------------------------------------------------------------------------
# Requires: streamlit, plotly, pandas, pillow, streamlit-plotly-events
#
# pip install streamlit==1.37.0 plotly==5.23.0 pandas pillow streamlit-plotly-events

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.graph_objs as go

import streamlit as st
from streamlit_plotly_events import plotly_events


# ============== CONFIG: all tabs (CSV paths + labels) =================
# These are the CSVs produced by your sweep driver.
# If a CSV is missing, the tab will show a friendly message.
TABS: Dict[str, Dict[str, str]] = {
    # DAB3_22_eq_results — Lm=10
    "SPS (Lm=10)":   {"csv": r"data\DAB3_22_eq_results\Lm_10\SPS_Lm10_hover_list.csv"},
    "Buck (Lm=10)":  {"csv": r"data\DAB3_22_eq_results\Lm_10\Buck_Lm10_hover_list.csv"},
    "Boost (Lm=10)": {"csv": r"data\DAB3_22_eq_results\Lm_10\Boost_Lm10_hover_list.csv"},
    "TZM (Lm=10)":   {"csv": r"data\DAB3_22_eq_results\Lm_10\TZM_Lm10_hover_list.csv"},

    # DAB3_22_eq_results — Lm=1e6
    "SPS (Lm=1e6)":   {"csv": r"data\DAB3_22_eq_results\Lm_1e+06\SPS_Lm1e+06_hover_list.csv"},
    "Buck (Lm=1e6)":  {"csv": r"data\DAB3_22_eq_results\Lm_1e+06\Buck_Lm1e+06_hover_list.csv"},
    "Boost (Lm=1e6)": {"csv": r"data\DAB3_22_eq_results\Lm_1e+06\Boost_Lm1e+06_hover_list.csv"},
    "TZM (Lm=1e6)":   {"csv": r"data\DAB3_22_eq_results\Lm_1e+06\TZM_Lm1e+06_hover_list.csv"},

    # LUTs from different folders
    "2-2 LUT (Lm=10)":  {"csv": r"data\DAB3_22_results\Lm_10\3-3 level_Lm10_hover_list.csv"},
    "2-2 LUT (Lm=1e6)": {"csv": r"data\DAB3_22_results\Lm_1e+06\3-3 level_Lm1e+06_hover_list.csv"},

    "3-2 LUT (Lm=10)":  {"csv": r"data\DAB3_32_results\Lm_10\3-3 level_Lm10_hover_list.csv"},
    "3-2 LUT (Lm=1e6)": {"csv": r"data\DAB3_32_results\Lm_1e+06\3-3 level_Lm1e+06_hover_list.csv"},

    "3-3 LUT (Lm=10)":  {"csv": r"data\DAB3_33_results\Lm_10\3-3 level_Lm10_hover_list.csv"},
    "3-3 LUT (Lm=1e6)": {"csv": r"data\DAB3_33_results\Lm_1e+06\3-3 level_Lm1e+06_hover_list.csv"},
}

# Fixed axes limits (like your MATLAB call { [0 1.5], [0 1.5] })
X_LIM = (0.0, 1.5)   # d
Y_LIM = (0.0, 1.5)   # P_calc


# ============== Helpers ==============

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have consistent column names:
      - d: 'd'
      - P_calc: prefer 'P_calc', else 'P', else 'Pset' as fallback
      - ZVS flags: 'ZVS_pri', 'ZVS_sec'
      - thumb path: 'thumb_path'
    """
    cols = {c.lower(): c for c in df.columns}
    # d
    dcol = None
    for name in ['d', 'd_vec', 'dval', 'gain', 'duty']:
        if name in cols:
            dcol = cols[name]; break
    # P_calc
    pcol = None
    for name in ['p_calc', 'pcalc', 'p', 'p_set', 'pset', 'p_req']:
        if name in cols:
            pcol = cols[name]; break
    # ZVS flags
    zpri = None
    for name in ['zvs_pri', 'zvspri', 'zpri']:
        if name in cols: zpri = cols[name]; break
    zsec = None
    for name in ['zvs_sec', 'zvssec', 'zsec']:
        if name in cols: zsec = cols[name]; break
    # thumb path
    tcol = None
    for name in ['thumb_path', 'thumb', 'png', 'png_path', 'image', 'image_path']:
        if name in cols: tcol = cols[name]; break

    needed = [dcol, pcol, zpri, zsec, tcol]
    if any(v is None for v in needed):
        missing = []
        if dcol  is None: missing.append('d')
        if pcol  is None: missing.append('P_calc (or P/Pset)')
        if zpri  is None: missing.append('ZVS_pri')
        if zsec  is None: missing.append('ZVS_sec')
        if tcol  is None: missing.append('thumb_path')
        raise ValueError(f"CSV missing columns: {', '.join(missing)}")

    out = pd.DataFrame({
        'd': df[dcol].astype(float),
        'P_calc': df[pcol].astype(float),
        'ZVS_pri': df[zpri].astype(float),
        'ZVS_sec': df[zsec].astype(float),
        'thumb_path': df[tcol].astype(str),
    })
    return out


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return normalize_columns(df)
    except Exception as e:
        st.info(f"Could not load **{path}**\n\n> {e}")
        return None


def split_masks(df: pd.DataFrame):
    """Return boolean masks for OK / HS_primary / HS_secondary / both."""
    pri_bad = df['ZVS_pri'] == 0
    sec_bad = df['ZVS_sec'] == 0
    both    = pri_bad & sec_bad
    pri     = pri_bad & ~sec_bad
    sec     = sec_bad & ~pri_bad
    ok      = ~(pri | sec | both)
    return ok, pri, sec, both


def blank_image(w=960, h=720) -> Image.Image:
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((16, 16), "Click a point to preview the PNG", fill=(60, 60, 60))
    return img


def build_left_plot(df: pd.DataFrame, highlight_idx: Optional[int] = None) -> go.Figure:
    """Plot P_calc vs d with color groups and optional highlighted point."""
    ok, pri, sec, both = split_masks(df)

    fig = go.Figure()

    # Teal OK
    fig.add_trace(go.Scattergl(
        x=df.loc[ok, 'd'], y=df.loc[ok, 'P_calc'],
        mode='markers', name='ZVS (both)',
        marker=dict(size=7, color='rgb(51,115,217)'),
        hovertemplate='d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>',
    ))
    # light grey HS_primary
    fig.add_trace(go.Scattergl(
        x=df.loc[pri, 'd'], y=df.loc[pri, 'P_calc'],
        mode='markers', name='HS_{primary}',
        marker=dict(size=7, color='rgb(204,204,204)'),
        hovertemplate='d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>',
    ))
    # dark grey HS_secondary
    fig.add_trace(go.Scattergl(
        x=df.loc[sec, 'd'], y=df.loc[sec, 'P_calc'],
        mode='markers', name='HS_{secondary}',
        marker=dict(size=7, color='rgb(102,102,102)'),
        hovertemplate='d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>',
    ))
    # charcoal both
    fig.add_trace(go.Scattergl(
        x=df.loc[both, 'd'], y=df.loc[both, 'P_calc'],
        mode='markers', name='HS both',
        marker=dict(size=8, color='rgb(26,26,26)'),
        hovertemplate='d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>',
    ))

    # Optional highlighted point (preview)
    if highlight_idx is not None and 0 <= highlight_idx < len(df):
        fig.add_trace(go.Scattergl(
            x=[df.loc[highlight_idx, 'd']], y=[df.loc[highlight_idx, 'P_calc']],
            mode='markers', name='(Preview)',
            marker=dict(size=12, symbol='circle-open', line=dict(width=2, color='red')),
            hoverinfo='skip'
        ))

    fig.update_layout(
        xaxis=dict(title='d', range=[X_LIM[0], X_LIM[1]], constrain='domain'),
        yaxis=dict(title='P_{calc}', range=[Y_LIM[0], Y_LIM[1]]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode='lasso',
        height=520,
    )
    return fig


def pick_clicked_index(clicks, df: pd.DataFrame) -> Optional[int]:
    """
    Convert plotly click result to dataframe index.
    We match on (x,y) to nearest row.
    """
    if not clicks:
        return None
    p = clicks[0]
    if 'x' not in p or 'y' not in p:
        return None
    x, y = p['x'], p['y']
    # nearest row in euclidean distance
    arr = (df['d'].to_numpy() - x) ** 2 + (df['P_calc'].to_numpy() - y) ** 2
    idx = int(np.argmin(arr))
    return idx


def show_right_preview(df: pd.DataFrame, idx: Optional[int]):
    st.markdown("**Preview**")
    if idx is None:
        st.image(blank_image(), use_container_width=True)
        return
    pth = df.loc[idx, 'thumb_path']
    if isinstance(pth, str) and len(pth) > 0 and os.path.isfile(pth):
        try:
            st.image(pth, use_container_width=True)
            st.caption(os.path.basename(pth))
        except Exception as e:
            st.image(blank_image(), use_container_width=True)
            st.info(f"Failed to open image:\n\n> {e}")
    else:
        st.image(blank_image(), use_container_width=True)
        st.info("No PNG found for this point.")


# ============== App ==============

st.set_page_config(page_title="Hover PNG Viewer (CSV)", layout="wide")
st.title("Hover PNG Viewer — CSV Tabs")

# Keep selected index per tab in session_state
if 'sel_idx' not in st.session_state:
    st.session_state['sel_idx'] = {}

tab_objs = st.tabs(list(TABS.keys()))

for tab_name, tab in zip(TABS.keys(), tab_objs):
    with tab:
        csv_path = TABS[tab_name]['csv']
        st.write(f"`{csv_path}`")

        df = load_csv_safe(csv_path)
        if df is None or df.empty:
            st.warning("No data to display.")
            continue

        # Apply fixed XY limits by filtering to the visible window (optional; still plot full but limits fixed)
        # Not filtering to keep full dataset; limits are enforced in the figure.

        # current selected index for this tab
        sel_idx = st.session_state['sel_idx'].get(tab_name)

        c1, c2 = st.columns([0.55, 0.45], gap="large")

        with c1:
            fig = build_left_plot(df, highlight_idx=sel_idx)
            clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=520, override_width="100%")
            new_idx = pick_clicked_index(clicks, df)
            if new_idx is not None:
                st.session_state['sel_idx'][tab_name] = new_idx
                sel_idx = new_idx

        with c2:
            show_right_preview(df, sel_idx)

st.caption("Teal: ZVS OK • Light grey: HS_primary • Dark grey: HS_secondary • Charcoal: both • Red ring: (Preview)")
