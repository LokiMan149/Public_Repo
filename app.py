# app.py — single Plotly figure (no duplicate), PNG preview updates on hover/click

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageOps
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Hover PNG Viewer — CSV Tabs", layout="wide")
BASE = Path(__file__).parent

def rel(*parts) -> Path:
    return BASE.joinpath(*parts)

def norm_from_csv(p: str) -> Path:
    pp = Path(str(p).replace("\\", "/"))
    return pp if pp.is_absolute() else BASE.joinpath(pp)

def load_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        if "thumb_path" in df.columns:
            df["thumb_path"] = (
                df["thumb_path"]
                .astype(str)
                .map(lambda s: str(norm_from_csv(s)))
            )
        return df
    except Exception as e:
        st.error(f"Could not load `{csv_path.as_posix()}`\n\n**{e}**")
        return pd.DataFrame()

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["d", "P_calc", "ZVS_pri", "ZVS_sec", "thumb_path"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -------- CONFIG: tabs & fixed axis limits ----------
TABS: Dict[str, Dict[str, Path]] = {
    # DAB3_22_eq_results — Lm=10
    "SPS (Lm=10)":   {"csv": rel("data","DAB3_22_eq_results","Lm_10","SPS_Lm10_hover_list.csv")},
    "Buck (Lm=10)":  {"csv": rel("data","DAB3_22_eq_results","Lm_10","Buck_Lm10_hover_list.csv")},
    "Boost (Lm=10)": {"csv": rel("data","DAB3_22_eq_results","Lm_10","Boost_Lm10_hover_list.csv")},
    "TZM (Lm=10)":   {"csv": rel("data","DAB3_22_eq_results","Lm_10","TZM_Lm10_hover_list.csv")},

    # DAB3_22_eq_results — Lm=1e6
    "SPS (Lm=1e6)":   {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","SPS_Lm1e+06_hover_list.csv")},
    "Buck (Lm=1e6)":  {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","Buck_Lm1e+06_hover_list.csv")},
    "Boost (Lm=1e6)": {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","Boost_Lm1e+06_hover_list.csv")},
    "TZM (Lm=1e6)":   {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","TZM_Lm1e+06_hover_list.csv")},

    # LUTs (put these files in your repo under data/)
    "2-2 LUT (Lm=10)":  {"csv": rel("data","DAB3_22_results","Lm_10","2-2 LUT_Lm10_hover_list.csv")},
    "2-2 LUT (Lm=1e6)": {"csv": rel("data","DAB3_22_results","Lm_1e+06","2-2 LUT_Lm1e+06_hover_list.csv")},
    "3-2 LUT (Lm=10)":  {"csv": rel("data","DAB3_32_results","Lm_10","3-2 level_Lm10_hover_list.csv")},
    "3-2 LUT (Lm=1e6)": {"csv": rel("data","DAB3_32_results","Lm_1e+06","3-2 level_Lm1e+06_hover_list.csv")},
    "3-3 LUT (Lm=10)":  {"csv": rel("data","DAB3_33_results","Lm_10","3-3 level_Lm10_hover_list.csv")},
    "3-3 LUT (Lm=1e6)": {"csv": rel("data","DAB3_33_results","Lm_1e+06","3-3 level_Lm1e+06_hover_list.csv")},
}

# Fixed axis limits like your MATLAB call { [0 1.5], [0 1.5] }
X_LIM: Tuple[float, float] = (0.0, 1.5)  # d
Y_LIM: Tuple[float, float] = (0.0, 1.5)  # P_calc

def build_scatter(df: pd.DataFrame, title: str, pre_x: Optional[float], pre_y: Optional[float]) -> go.Figure:
    df = ensure_columns(df).copy()

    valid = df["d"].notna() & df["P_calc"].notna()
    pri_bad = (df["ZVS_pri"] == 0) & valid
    sec_bad = (df["ZVS_sec"] == 0) & valid
    both_bad = pri_bad & sec_bad
    pri_only = pri_bad & ~sec_bad
    sec_only = sec_bad & ~pri_bad
    ok = valid & ~(pri_bad | sec_bad)

    def mk(mask: pd.Series, name: str, color: str, size: int = 6):
        dd = df.loc[mask]  # correct: brackets
        return go.Scatter(
            x=dd["d"], y=dd["P_calc"],
            mode="markers", name=name,
            marker=dict(color=color, size=size, opacity=0.7),
            customdata=np.stack([dd.get("thumb_path", pd.Series([""] * len(dd)))], axis=1),
            hovertemplate="d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>",
        )

    c_ok, c_pri, c_sec, c_both = "rgb(51,115,217)", "rgb(204,204,204)", "rgb(102,102,102)", "rgb(25,25,25)"

    fig = go.Figure()
    fig.add_trace(mk(ok,       "ZVS (both)",     c_ok))
    fig.add_trace(mk(pri_only, "HS_{primary}",   c_pri))
    fig.add_trace(mk(sec_only, "HS_{secondary}", c_sec))
    fig.add_trace(mk(both_bad, "HS both",        c_both))

    # preview point (red ring)
    px = pre_x if pre_x is not None else np.nan
    py = pre_y if pre_y is not None else np.nan
    fig.add_trace(go.Scatter(
        x=[px], y=[py], mode="markers", name="(Preview)",
        marker=dict(color="rgb(214,69,65)", size=8, line=dict(width=1, color="black")),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="d", range=X_LIM),
        yaxis=dict(title="P_{calc}", range=Y_LIM),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        height=640,
    )
    return fig

def draw_preview(png_path: Optional[str]) -> Image.Image:
    W, H = 1280, 800
    if png_path:
        p = norm_from_csv(png_path)
        if p.is_file():
            try:
                im = Image.open(p).convert("RGB")
                return ImageOps.contain(im, (W, H))
            except Exception:
                pass
    # fallback checkerboard
    tile = Image.new("RGB", (16, 16), "white")
    alt  = Image.new("RGB", (16, 16), "black")
    row = Image.new("RGB", (16 * 40, 16))
    for i in range(40):
        row.paste(tile if i % 2 == 0 else alt, (i * 16, 0))
    board = Image.new("RGB", (16 * 40, 16 * 25))
    for j in range(25):
        board.paste(row if j % 2 == 0 else ImageOps.invert(row), (0, j * 16))
    return ImageOps.contain(board, (W, H))

st.markdown("# Hover PNG Viewer — CSV Tabs")

tab_objs = st.tabs(list(TABS.keys()))
for tab_name, st_tab in zip(TABS.keys(), tab_objs):
    with st_tab:
        csv_path = TABS[tab_name]["csv"]
        st.caption(csv_path.as_posix())

        df = load_csv(csv_path)
        if df.empty:
            st.info("No data to display.")
            continue

        state_key = f"sel_{tab_name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {"x": None, "y": None, "png": None}
        sel = st.session_state[state_key]

        fig = build_scatter(df, title="", pre_x=sel["x"], pre_y=sel["y"])

        # Lay out: left = single plot (rendered by plotly_events), right = PNG preview
        col_plot, col_png = st.columns((1.05, 1.35), gap="large")

        with col_plot:
            events = plotly_events(
                fig,
                click_event=True,
                hover_event=True,
                select_event=False,
                override_height=640,
                key=f"plt_{tab_name}",
            )

        # update selection from last event (hover or click)
        if events:
            ev = events[-1]
            sel["x"] = ev.get("x", sel["x"])
            sel["y"] = ev.get("y", sel["y"])
            # customdata may come as list/np array nested like [["/path.png"]]
            cd = ev.get("customdata", None)
            if cd is not None:
                try:
                    # list -> first element; ndarray -> index [0]
                    val = cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd
                    sel["png"] = val
                except Exception:
                    pass

        with col_png:
            st.image(draw_preview(sel["png"]), caption=(Path(sel["png"]).name if sel["png"] else "(no PNG)"))

st.markdown(
    "<span style='color:#3373D9;'>Teal</span>: ZVS OK &nbsp; • &nbsp; "
    "<span style='color:#CCCCCC;'>Light grey</span>: HS_primary &nbsp; • &nbsp; "
    "<span style='color:#666666;'>Dark grey</span>: HS_secondary &nbsp; • &nbsp; "
    "<span style='color:#191919;'>Charcoal</span>: both &nbsp; • &nbsp; "
    "<span style='color:#D64541;'>Red ring</span>: (Preview)",
    unsafe_allow_html=True,
)
