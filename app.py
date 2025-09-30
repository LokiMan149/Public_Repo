# app.py
# Streamlit hover PNG viewer with per-tab CSVs and global/per-tab axis limits.
import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Hover Map Viewer", layout="wide")

# ---------------------------- CONFIG: TABS -----------------------------------
# Edit paths to match your repo. Paths can be local (relative to this file) or absolute URLs in your CSVs.
TABS = [
    # Lm = 10
    {"key":"sps10",   "name":"SPS (Lm=10)",   "csv":"data/DAB3_22_results/Lm_10/SPS_Lm10_hover_list.csv"},
    {"key":"buck10",  "name":"Buck (Lm=10)",  "csv":"data/DAB3_22_results/Lm_10/Buck_Lm10_hover_list.csv"},
    {"key":"boost10", "name":"Boost (Lm=10)", "csv":"data/DAB3_22_results/Lm_10/Boost_Lm10_hover_list.csv"},
    {"key":"tzm10",   "name":"TZM (Lm=10)",   "csv":"data/DAB3_22_results/Lm_10/TZM_Lm10_hover_list.csv"},
    # Lm = 1e6
    {"key":"sps1e6",   "name":"SPS (Lm=1e6)",   "csv":"data/DAB3_22_results/Lm_1e+06/SPS_Lm1e+06_hover_list.csv"},
    {"key":"buck1e6",  "name":"Buck (Lm=1e6)",  "csv":"data/DAB3_22_results/Lm_1e+06/Buck_Lm1e+06_hover_list.csv"},
    {"key":"boost1e6", "name":"Boost (Lm=1e6)", "csv":"data/DAB3_22_results/Lm_1e+06/Boost_Lm1e+06_hover_list.csv"},
    {"key":"tzm1e6",   "name":"TZM (Lm=1e6)",   "csv":"data/DAB3_22_results/Lm_1e+06/TZM_Lm1e+06_hover_list.csv"},
    # LUTs (2-2 and 3-3) — change CSV names if yours differ
    {"key":"lut22_10",   "name":"2-2 LUT (Lm=10)",   "csv":"data/DAB3_22_results/Lm_10/2-2 LUT_Lm10_hover_list.csv"},
    {"key":"lut22_1e6",  "name":"2-2 LUT (Lm=1e6)",  "csv":"data/DAB3_22_results/Lm_1e+06/2-2 LUT_Lm1e+06_hover_list.csv"},
    {"key":"lut33_10",   "name":"3-3 LUT (Lm=10)",   "csv":"data/DAB3_33_results/Lm_10/3-3 level_Lm10_hover_list.csv"},
    {"key":"lut33_1e6",  "name":"3-3 LUT (Lm=1e6)",  "csv":"data/DAB3_33_results/Lm_1e+06/3-3 level_Lm1e+06_hover_list.csv"},
]

# Colors (match your MATLAB style)
COL_OK   = "rgb(51,120,217)"   # teal-ish
COL_PRI  = "rgb(204,204,204)"  # light gray
COL_SEC  = "rgb(102,102,102)"  # dark gray
COL_BOTH = "rgb(26,26,26)"     # charcoal
COL_PREV = "rgb(220,60,30)"    # red

# -------------------------- HELPERS & CACHING --------------------------------
def checkerboard(w=1280, h=960, cell=20):
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x//cell) + (y//cell)) % 2 == 0:
                draw.rectangle([x, y, x+cell, y+cell], fill=(230,230,230))
            else:
                draw.rectangle([x, y, x+cell, y+cell], fill=(20,20,20))
    return img

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    # Read and normalize column names across variants
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    def col(*names):
        for n in names:
            k = n.lower()
            if k in cols:
                return cols[k]
        raise KeyError(f"Missing any of columns {names} in {path}")

    # Try common variants
    d_col       = col("d")
    pcalc_col   = col("p_calc", "Pcalc", "P_actual", "P")  # flexible
    zpri_col    = col("zvs_pri","ZVS_pri")
    zsec_col    = col("zvs_sec","ZVS_sec")
    thumb_col   = col("thumb_path","png","preview","image")

    df = df[[d_col, pcalc_col, zpri_col, zsec_col, thumb_col]].copy()
    df.columns = ["d", "P_calc", "ZVS_pri", "ZVS_sec", "thumb_path"]

    # Normalize booleans: 0=hard-switch, 1=ZVS
    df["ZVS_pri"] = pd.to_numeric(df["ZVS_pri"], errors="coerce")
    df["ZVS_sec"] = pd.to_numeric(df["ZVS_sec"], errors="coerce")

    # Make sure paths are strings
    df["thumb_path"] = df["thumb_path"].astype(str)
    return df

def resolve_image(path_or_url: str) -> Image.Image:
    # Local file?
    if "://" not in path_or_url:
        p = Path(path_or_url)
        if not p.is_file():
            # try relative to app directory
            p = Path(__file__).parent / path_or_url
        if p.is_file():
            return Image.open(p)
        # fallback checkerboard if missing
        return checkerboard(1280, 960, 24)
    # Remote URL
    try:
        import requests
        r = requests.get(path_or_url, timeout=8)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return checkerboard(1280, 960, 24)

@st.cache_data(show_spinner=False)
def compute_global_limits(dfs: dict):
    xs, ys = [], []
    for df in dfs.values():
        xs.append(df["d"].to_numpy(np.float64))
        ys.append(df["P_calc"].to_numpy(np.float64))
    if not xs or not ys:
        return (0,1), (0,1)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0,1), (0,1)
    return (float(np.nanmin(x)), float(np.nanmax(x))), (float(np.nanmin(y)), float(np.nanmax(y)))

def build_figure(df, xlim=None, ylim=None, prev_xy=None):
    # Masks like your MATLAB viewer
    valid = np.isfinite(df["d"]) & np.isfinite(df["P_calc"])
    zpri0 = (df["ZVS_pri"] == 0)
    zsec0 = (df["ZVS_sec"] == 0)

    idx_ok   = valid & ~(zpri0 | zsec0)
    idx_pri  = valid &  (zpri0 & ~zsec0)
    idx_sec  = valid &  (zsec0 & ~zpri0)
    idx_both = valid &  (zpri0 &  zsec0)

    fig = go.Figure()

    # order matters (we use curveNumber to map back)
    groups = [
        ("ZVS (both)",     COL_OK,   idx_ok),
        ("HS_{primary}",   COL_PRI,  idx_pri),
        ("HS_{secondary}", COL_SEC,  idx_sec),
        ("HS both",        COL_BOTH, idx_both),
    ]
    sizes = 6
    for name, col, m in groups:
        fig.add_trace(go.Scattergl(
            x=df.loc[m,"d"], y=df.loc[m,"P_calc"],
            mode="markers",
            name=name,
            marker=dict(color=col, size=sizes),
            hoverinfo="skip"
        ))

    # Preview marker (red) — single point
    if prev_xy is not None:
        fig.add_trace(go.Scatter(
            x=[prev_xy[0]], y=[prev_xy[1]],
            mode="markers",
            name="(Preview)", marker=dict(color=COL_PREV, size=9, line=dict(color="black", width=1))
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers", name="(Preview)",
            marker=dict(color=COL_PREV, size=9, line=dict(color="black", width=1))
        ))

    fig.update_layout(
        height=740,
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="d", yaxis_title="P_{calc}"
    )
    if xlim is not None and xlim[1] > xlim[0]:
        fig.update_xaxes(range=list(xlim))
    if ylim is not None and ylim[1] > ylim[0]:
        fig.update_yaxes(range=list(ylim))
    return fig

# ------------------------------ SIDEBAR UI -----------------------------------
st.sidebar.title("Axes limits")
mode = st.sidebar.radio("Mode:", ["Per tab (auto)","Global (auto)","Manual"], index=1)

# Pre-load all CSVs to enable global limits
loaded = {}
errors = []
for t in TABS:
    try:
        loaded[t["key"]] = load_csv(t["csv"])
    except Exception as e:
        errors.append(f'Error loading: {t["csv"]} — {e}')

if errors:
    st.sidebar.warning("\n\n".join(errors))

xlim_g, ylim_g = compute_global_limits(loaded) if loaded else ((0,1),(0,1))
if mode == "Manual":
    xmin = st.sidebar.number_input("xmin", value=float(xlim_g[0]))
    xmax = st.sidebar.number_input("xmax", value=float(xlim_g[1]))
    ymin = st.sidebar.number_input("ymin", value=float(ylim_g[0]))
    ymax = st.sidebar.number_input("ymax", value=float(ylim_g[1]))
    xlim_sel = (xmin, xmax)
    ylim_sel = (ymin, ymax)
elif mode == "Global (auto)":
    xlim_sel, ylim_sel = xlim_g, ylim_g
else:
    xlim_sel, ylim_sel = None, None  # per-tab later

# ------------------------------- MAIN TABS -----------------------------------
tabs = st.tabs([t["name"] for t in TABS])

# Keep per-tab selection state
if "sel" not in st.session_state:
    st.session_state.sel = {}  # key -> dict(path, xy)

for t, tab in zip(TABS, tabs):
    key = t["key"]
    with tab:
        if key not in loaded:
            st.error(f"Could not load {t['csv']}")
            continue

        df = loaded[key]

        # Axis limits for this tab
        if mode == "Per tab (auto)":
            xlim_l = (float(df["d"].min()), float(df["d"].max()))
            ylim_l = (float(df["P_calc"].min()), float(df["P_calc"].max()))
        else:
            xlim_l, ylim_l = xlim_sel, ylim_sel

        # Current preview (persist across reruns)
        prev = st.session_state.sel.get(key, {"path":"", "xy":None})
        prev_xy = prev["xy"]

        c1, c2 = st.columns([1, 1.35], gap="large")

        with c1:
            fig = build_figure(df, xlim=xlim_l, ylim=ylim_l, prev_xy=prev_xy)
            events = plotly_events(
                fig, click_event=True, hover_event=True, select_event=False,
                override_height=740, key=f"pe-{key}"
            )

            # Map click to a data row -> image path
            # Trace order: 0 ok, 1 pri, 2 sec, 3 both, 4 preview
            if events:
                ev = events[0]
                cn = ev.get("curveNumber", -1)
                pi = ev.get("pointIndex", -1)
                if 0 <= cn <= 3 and 0 <= pi:
                    # Build masks like in build_figure
                    valid = np.isfinite(df["d"]) & np.isfinite(df["P_calc"])
                    zpri0 = (df["ZVS_pri"] == 0)
                    zsec0 = (df["ZVS_sec"] == 0)
                    idx_ok   = valid & ~(zpri0 | zsec0)
                    idx_pri  = valid &  (zpri0 & ~zsec0)
                    idx_sec  = valid &  (zsec0 & ~zpri0)
                    idx_both = valid &  (zpri0 &  zsec0)
                    idx_lists = [
                        list(df.index[idx_ok]),
                        list(df.index[idx_pri]),
                        list(df.index[idx_sec]),
                        list(df.index[idx_both]),
                    ]
                    if cn < len(idx_lists) and pi < len(idx_lists[cn]):
                        ridx = idx_lists[cn][pi]
                        pth  = df.at[ridx, "thumb_path"]
                        xy   = (float(df.at[ridx,"d"]), float(df.at[ridx,"P_calc"]))
                        st.session_state.sel[key] = {"path": pth, "xy": xy}
                        prev = st.session_state.sel[key]
                        prev_xy = xy

        with c2:
            # Display selected or checkerboard
            pth = st.session_state.sel.get(key, {}).get("path", "")
            if pth:
                img = resolve_image(pth)
                st.image(img, use_column_width=True)
                st.caption(os.path.basename(pth))
            else:
                st.image(checkerboard(1280, 960, 24), use_column_width=True)
                st.caption("Hover/click a point to preview")

# Footer
st.markdown(
    "<div style='text-align:right;color:#999'>Streamlit Hover Map Viewer</div>",
    unsafe_allow_html=True
)
