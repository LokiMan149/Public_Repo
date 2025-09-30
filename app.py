from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageOps
import streamlit as st
from streamlit_plotly_events import plotly_events

# ----------------- Page config -----------------
st.set_page_config(page_title="Hover PNG Viewer — CSV Tabs", layout="wide")
BASE = Path(__file__).parent.resolve()

# ----------------- Helpers -----------------
def rel(*parts) -> Path:
    return BASE.joinpath(*parts)

def norm_path(p: str | Path) -> Path:
    p = Path(str(p).replace("\\", "/"))
    return p if p.is_absolute() else BASE.joinpath(p)

@st.cache_data(show_spinner=False)
def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize expected columns
    for c in ["d", "P_calc", "ZVS_pri", "ZVS_sec", "thumb_path"]:
        if c not in df.columns:
            df[c] = np.nan
    df["thumb_path"] = df["thumb_path"].astype(str).map(lambda s: str(norm_path(s)) if s else "")
    return df

def png_preview(path: Optional[str]) -> Image.Image:
    W, H = 1280, 800
    if path:
        p = norm_path(path)
        if p.is_file():
            try:
                im = Image.open(p).convert("RGB")
                return ImageOps.contain(im, (W, H))
            except Exception:
                pass
    # checkerboard fallback
    tile = Image.new("RGB", (16, 16), "white")
    alt  = Image.new("RGB", (16, 16), "black")
    row = Image.new("RGB", (16 * 40, 16))
    for i in range(40):
        row.paste(tile if i % 2 == 0 else alt, (i * 16, 0))
    board = Image.new("RGB", (16 * 40, 16 * 25))
    for j in range(25):
        board.paste(row if j % 2 == 0 else ImageOps.invert(row), (0, j * 16))
    return ImageOps.contain(board, (W, H))

def build_fig(df: pd.DataFrame, pre_xy: Tuple[Optional[float], Optional[float]]) -> go.Figure:
    # masks
    valid   = df["d"].notna() & df["P_calc"].notna()
    pri_bad = (df["ZVS_pri"] == 0) & valid
    sec_bad = (df["ZVS_sec"] == 0) & valid
    both    = pri_bad & sec_bad
    pri     = pri_bad & ~sec_bad
    sec     = sec_bad & ~pri_bad
    ok      = valid & ~(pri_bad | sec_bad)

    def mk(mask: pd.Series, name: str, color: str, size: int = 8):
        dd = df.loc[mask]
        # customdata as 1-col array with PNG path
        cd = np.stack([dd["thumb_path"].to_numpy()], axis=1) if len(dd) else np.empty((0,1))
        return go.Scatter(
            x=dd["d"],
            y=dd["P_calc"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, opacity=0.85),
            customdata=cd,
            hovertemplate="d=%{x:.4g}<br>P=%{y:.4g}<extra></extra>",
        )

    c_ok, c_pri, c_sec, c_both = "rgb(51,115,217)", "rgb(204,204,204)", "rgb(102,102,102)", "rgb(25,25,25)"

    fig = go.Figure()
    fig.add_trace(mk(ok,   "ZVS (both)",     c_ok))
    fig.add_trace(mk(pri,  "HS_{primary}",   c_pri))
    fig.add_trace(mk(sec,  "HS_{secondary}", c_sec))
    fig.add_trace(mk(both, "HS both",        c_both))

    # preview dot (red ring)
    px, py = pre_xy
    fig.add_trace(go.Scatter(
        x=[px if px is not None else np.nan],
        y=[py if py is not None else np.nan],
        mode="markers",
        name="(Preview)",
        marker=dict(color="rgba(214,69,65,0.0)", size=12, line=dict(width=2, color="rgba(214,69,65,1.0)")),
        hoverinfo="skip",
        showlegend=True,
    ))

    fig.update_layout(
        dragmode="pan",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        height=640,
        xaxis=dict(title="d",  range=[0.0, 1.5]),
        yaxis=dict(title="P_{calc}", range=[0.0, 1.5]),
    )
    return fig

def parse_customdata(ev) -> Optional[str]:
    cd = ev.get("customdata", None)
    if cd is None:
        return None
    # cd may be ['path'] or [['path']] or np.array([...])
    try:
        if isinstance(cd, (list, tuple, np.ndarray)):
            first = cd[0] if len(cd) else None
            if isinstance(first, (list, tuple, np.ndarray)):
                return str(first[0]) if len(first) else None
            return str(first)
        return str(cd)
    except Exception:
        return None

def nearest_point(df: pd.DataFrame, x: float, y: float) -> Optional[pd.Series]:
    """If plotly doesn't return customdata, snap to nearest valid point."""
    valid = df["d"].notna() & df["P_calc"].notna()
    if not valid.any():
        return None
    dx = df.loc[valid, "d"].to_numpy() - x
    dy = df.loc[valid, "P_calc"].to_numpy() - y
    i  = int(np.argmin(dx*dx + dy*dy))
    return df.loc[valid].iloc[i]

# ----------------- CONFIG: tabs -----------------
TABS: Dict[str, Dict[str, Path]] = {
    # 22_eq, Lm=10
    "SPS (Lm=10)":   {"csv": rel("data","DAB3_22_eq_results","Lm_10","SPS_Lm10_hover_list.csv")},
    "Buck (Lm=10)":  {"csv": rel("data","DAB3_22_eq_results","Lm_10","Buck_Lm10_hover_list.csv")},
    "Boost (Lm=10)": {"csv": rel("data","DAB3_22_eq_results","Lm_10","Boost_Lm10_hover_list.csv")},
    "TZM (Lm=10)":   {"csv": rel("data","DAB3_22_eq_results","Lm_10","TZM_Lm10_hover_list.csv")},

    # 22_eq, Lm=1e6
    "SPS (Lm=1e6)":   {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","SPS_Lm1e+06_hover_list.csv")},
    "Buck (Lm=1e6)":  {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","Buck_Lm1e+06_hover_list.csv")},
    "Boost (Lm=1e6)": {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","Boost_Lm1e+06_hover_list.csv")},
    "TZM (Lm=1e6)":   {"csv": rel("data","DAB3_22_eq_results","Lm_1e+06","TZM_Lm1e+06_hover_list.csv")},

    # LUTs (make sure these exact files exist; Linux is case-sensitive)
    "2-2 LUT (Lm=10)":  {"csv": rel("data","DAB3_22_results","Lm_10","2-2 LUT_Lm10_hover_list.csv")},
    "2-2 LUT (Lm=1e6)": {"csv": rel("data","DAB3_22_results","Lm_1e+06","2-2 LUT_Lm1e+06_hover_list.csv")},
    "3-2 LUT (Lm=10)":  {"csv": rel("data","DAB3_32_results","Lm_10","3-2 level_Lm10_hover_list.csv")},
    "3-2 LUT (Lm=1e6)": {"csv": rel("data","DAB3_32_results","Lm_1e+06","3-2 level_Lm1e+06_hover_list.csv")},
    "3-3 LUT (Lm=10)":  {"csv": rel("data","DAB3_33_results","Lm_10","3-3 level_Lm10_hover_list.csv")},
    "3-3 LUT (Lm=1e6)": {"csv": rel("data","DAB3_33_results","Lm_1e+06","3-3 level_Lm1e+06_hover_list.csv")},
}

# ----------------- Sidebar diagnostics -----------------
with st.sidebar:
    st.markdown("### Diagnostics")
    st.write("Working dir:", str(BASE))
    data_dir = BASE / "data"
    st.write("`data/` exists:", data_dir.exists())
    if data_dir.exists():
        sample = [p.as_posix() for p in data_dir.rglob("*.csv")]
        st.write("CSV files Streamlit can see (first 30):")
        st.write(sample[:30])

st.markdown("# Hover PNG Viewer — CSV Tabs")

# ----------------- Tabs UI -----------------
tabs = st.tabs(list(TABS.keys()))

for tab_name, tab in zip(TABS.keys(), tabs):
    with tab:
        csv_path = TABS[tab_name]["csv"]
        st.caption(csv_path.as_posix())

        if not csv_path.is_file():
            st.error(f"Could not load `{csv_path.as_posix()}` (file not found)")
            # Show parent directory listing to quickly spot naming/case/space issues
            parent = csv_path.parent
            st.info(f"Contents of `{parent.as_posix()}`:")
            if parent.exists():
                listing = [p.name for p in sorted(parent.iterdir())]
                st.write(listing)
            else:
                st.write("(Parent directory does not exist)")
            continue

        try:
            df = load_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            continue

        # init state
        state_key = f"sel::{tab_name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {"x": None, "y": None, "png": None}
        sel = st.session_state[state_key]

        fig = build_fig(df, (sel["x"], sel["y"]))

        # layout: left plot, right PNG
        left, right = st.columns((1.05, 1.35), gap="large")

        with left:
            events = plotly_events(
                fig,
                click_event=True,
                hover_event=True,
                select_event=False,
                override_height=640,
                key=f"plt::{tab_name}",
            )

        # Pick the last event (hover or click)
        if events:
            ev = events[-1]
            # Try to read customdata (PNG path)
            png = parse_customdata(ev)
            x = ev.get("x", None)
            y = ev.get("y", None)

            # Fallback: snap to nearest DF point if png not present
            if png is None and x is not None and y is not None:
                row = nearest_point(df, float(x), float(y))
                if row is not None:
                    png = row.get("thumb_path", None)
                    x = float(row["d"])
                    y = float(row["P_calc"])

            # Update state if we have coordinates
            if x is not None and y is not None:
                sel["x"], sel["y"], sel["png"] = float(x), float(y), png

        with right:
            st.image(png_preview(sel["png"]), caption=(Path(sel["png"]).name if sel["png"] else "(no PNG)"))

st.markdown(
    "<span style='color:#3373D9;'>Teal</span>: ZVS OK &nbsp; • &nbsp; "
    "<span style='color:#CCCCCC;'>Light grey</span>: HS_primary &nbsp; • &nbsp; "
    "<span style='color:#666666;'>Dark grey</span>: HS_secondary &nbsp; • &nbsp; "
    "<span style='color:#191919;'>Charcoal</span>: both &nbsp; • &nbsp; "
    "<span style='color:#D64541;'>Red ring</span>: (Preview)",
    unsafe_allow_html=True,
)
