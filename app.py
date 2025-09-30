# app.py
# Streamlit hover PNG viewer (CSV tabs + fixed xy limits + cross-platform paths)

from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
import streamlit as st
from streamlit_plotly_events import plotly_events


# ------------------- page config -------------------
st.set_page_config(page_title="Hover PNG Viewer — CSV Tabs", layout="wide")

# ------------------- paths & helpers -------------------
BASE = Path(__file__).parent  # repo root for relative paths


def rel(*parts) -> Path:
    """Repo-relative path (portable)."""
    return BASE.joinpath(*parts)


def norm_from_csv(p: str) -> Path:
    """
    Normalize Windows backslashes coming from CSV and
    return a repo-relative absolute Path.
    """
    # Allow absolute paths too (but prefer repo-relative inputs)
    pp = Path(p.replace("\\", "/"))
    if pp.is_absolute():
        return pp
    return BASE.joinpath(pp)


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read a hover CSV; return empty df on failure with reason in session state."""
    try:
        df = pd.read_csv(csv_path)
        # normalize path column if present
        if "thumb_path" in df.columns:
            df["thumb_path"] = df["thumb_path"].astype(str).map(lambda s: str(norm_from_csv(s)))
        return df
    except Exception as e:
        st.error(f"Could not load `{csv_path.as_posix()}`\n\n**{e}**")
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected columns exist; create if missing, so the app keeps running."""
    needed = ["d", "P_calc", "ZVS_pri", "ZVS_sec", "thumb_path"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ------------------- CONFIG: tabs + fixed limits -------------------
# Keep these relative (forward slashes) so they work on Streamlit Cloud too.
TABS: Dict[str, Dict[str, Path]] = {
    # DAB3_22_eq_results — Lm=10
    "SPS (Lm=10)":   {"csv": rel("data", "DAB3_22_eq_results", "Lm_10",   "SPS_Lm10_hover_list.csv")},
    "Buck (Lm=10)":  {"csv": rel("data", "DAB3_22_eq_results", "Lm_10",   "Buck_Lm10_hover_list.csv")},
    "Boost (Lm=10)": {"csv": rel("data", "DAB3_22_eq_results", "Lm_10",   "Boost_Lm10_hover_list.csv")},
    "TZM (Lm=10)":   {"csv": rel("data", "DAB3_22_eq_results", "Lm_10",   "TZM_Lm10_hover_list.csv")},

    # DAB3_22_eq_results — Lm=1e6
    "SPS (Lm=1e6)":   {"csv": rel("data", "DAB3_22_eq_results", "Lm_1e+06", "SPS_Lm1e+06_hover_list.csv")},
    "Buck (Lm=1e6)":  {"csv": rel("data", "DAB3_22_eq_results", "Lm_1e+06", "Buck_Lm1e+06_hover_list.csv")},
    "Boost (Lm=1e6)": {"csv": rel("data", "DAB3_22_eq_results", "Lm_1e+06", "Boost_Lm1e+06_hover_list.csv")},
    "TZM (Lm=1e6)":   {"csv": rel("data", "DAB3_22_eq_results", "Lm_1e+06", "TZM_Lm1e+06_hover_list.csv")},

    # LUTs from different folders
    "2-2 LUT (Lm=10)":  {"csv": rel("data", "DAB3_22_results", "Lm_10",   "2-2 LUT_Lm10_hover_list.csv")},
    "2-2 LUT (Lm=1e6)": {"csv": rel("data", "DAB3_22_results", "Lm_1e+06","2-2 LUT_Lm1e+06_hover_list.csv")},

    "3-2 LUT (Lm=10)":  {"csv": rel("data", "DAB3_32_results", "Lm_10",   "3-2 level_Lm10_hover_list.csv")},
    "3-2 LUT (Lm=1e6)": {"csv": rel("data", "DAB3_32_results", "Lm_1e+06","3-2 level_Lm1e+06_hover_list.csv")},

    "3-3 LUT (Lm=10)":  {"csv": rel("data", "DAB3_33_results", "Lm_10",   "3-3 level_Lm10_hover_list.csv")},
    "3-3 LUT (Lm=1e6)": {"csv": rel("data", "DAB3_33_results", "Lm_1e+06","3-3 level_Lm1e+06_hover_list.csv")},
}

# Fixed axes limits (like your MATLAB call: { [0 1.5], [0 1.5] })
X_LIM: Tuple[float, float] = (0.0, 1.5)   # d
Y_LIM: Tuple[float, float] = (0.0, 1.5)   # P_calc


# ------------------- plot builder -------------------
def build_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    """
    Make the left scatter with categories:
      OK (both ZVS), HS_primary, HS_secondary, HS both (charcoal).
    """
    df = ensure_columns(df).copy()

    # Mask categories
    valid = df["d"].notna() & df["P_calc"].notna()
    pri_bad = (df["ZVS_pri"] == 0) & valid
    sec_bad = (df["ZVS_sec"] == 0) & valid
    both_bad = pri_bad & sec_bad
    pri_only = pri_bad & ~sec_bad
    sec_only = sec_bad & ~pri_bad
    ok = valid & ~(pri_bad | sec_bad)

    # Helper to create a trace
    def trace(mask, name, color, marker_size=6):
        dd = df.loc[mask]
        return go.Scatter(
            x=dd["d"], y=dd["P_calc"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=marker_size, opacity=0.7),
            customdata=np.stack([dd.get("thumb_path", pd.Series([""] * len(dd)))], axis=1),
            hovertemplate="d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>",
        )

    fig = go.Figure()

    # Colors matching your MATLAB scheme
    c_ok   = "rgb(51,115,217)"   # teal-ish blue
    c_pri  = "rgb(204,204,204)"  # light grey
    c_sec  = "rgb(102,102,102)"  # dark grey
    c_both = "rgb(25,25,25)"     # charcoal

    fig.add_trace(trace(ok,       "ZVS (both)",    c_ok))
    fig.add_trace(trace(pri_only, "HS_{primary}",  c_pri))
    fig.add_trace(trace(sec_only, "HS_{secondary}",c_sec))
    fig.add_trace(trace(both_bad, "HS both",       c_both))

    # Preview point (red), initially hidden (NaNs)
    fig.add_trace(go.Scatter(
        x=[np.nan], y=[np.nan],
        mode="markers",
        name="(Preview)",
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


def update_preview_marker(fig: go.Figure, x: float, y: float) -> go.Figure:
    """Move the red '(Preview)' marker to (x,y)."""
    # Red marker is the last trace
    fig.data[-1].x = [x]
    fig.data[-1].y = [y]
    return fig


# ------------------- preview image -------------------
def draw_preview(png_path: Optional[str]) -> Image.Image:
    """Open a PNG; return a checkerboard placeholder if not available."""
    W, H = 1280, 800  # preview canvas
    if png_path:
        p = norm_from_csv(png_path)
        if p.is_file():
            try:
                im = Image.open(p).convert("RGB")
                im = ImageOps.contain(im, (W, H))
                return im
            except Exception:
                pass
    # Checkerboard fallback
    tile = Image.new("RGB", (16, 16), "white")
    alt = Image.new("RGB", (16, 16), "black")
    row = Image.new("RGB", (16 * 40, 16))
    for i in range(40):
        row.paste(tile if i % 2 == 0 else alt, (i * 16, 0))
    board = Image.new("RGB", (16 * 40, 16 * 25))
    for j in range(25):
        board.paste(row if j % 2 == 0 else ImageOps.invert(row), (0, j * 16))
    return ImageOps.contain(board, (W, H))


# ------------------- UI -------------------
st.markdown("# Hover PNG Viewer — CSV Tabs")

tab_objs = st.tabs(list(TABS.keys()))
for tab_name, st_tab in zip(TABS.keys(), tab_objs):
    with st_tab:
        csv_path = TABS[tab_name]["csv"]
        st.caption(f"`{csv_path.as_posix()}`")

        df = load_csv(csv_path)
        if df.empty:
            st.info("No data to display.")
            continue

        fig = build_scatter(df, title="")
        # Handle plotly hover/click events
        #   - hover_event = {"x":..., "y":..., "curveNumber":..., "pointNumber":..., "customdata":[thumb_path]}
        events = plotly_events(
            fig,
            click_event=True,
            hover_event=True,
            select_event=False,
            override_height=640,
            override_width=None,
            key=f"plt_{tab_name}",
        )

        # Pick last event (hover prioritized over click if both)
        png_to_show = None
        px = py = None
        if events:
            ev = events[-1]
            px = ev.get("x", None)
            py = ev.get("y", None)
            cd = ev.get("customdata", None)
            if isinstance(cd, list) and len(cd) >= 1:
                png_to_show = cd[0]

        # Move the red preview marker on the client plot
        if px is not None and py is not None:
            fig = update_preview_marker(fig, px, py)

        col_scatter, col_png = st.columns((1.05, 1.35), gap="large")
        with col_scatter:
            # re-render with red marker update (no events capture)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_png:
            st.image(draw_preview(png_to_show), caption=Path(png_to_show).name if png_to_show else "(no PNG)")

# ------------- footer legend (colors) -------------
st.markdown(
    "<span style='color:#3373D9;'>Teal</span>: ZVS OK &nbsp; • &nbsp; "
    "<span style='color:#CCCCCC;'>Light grey</span>: HS_primary &nbsp; • &nbsp; "
    "<span style='color:#666666;'>Dark grey</span>: HS_secondary &nbsp; • &nbsp; "
    "<span style='color:#191919;'>Charcoal</span>: both &nbsp; • &nbsp; "
    "<span style='color:#D64541;'>Red ring</span>: (Preview)",
    unsafe_allow_html=True,
)
