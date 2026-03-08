"""AutoGateDetector — Streamlit results viewer.

Reads gates.json from the results/ directory (or AGATEDETECTOR_RESULTS env var).
Shows a filterable table, summary stats, and a 3D scatter of gate bboxes.
"""

import json
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AutoGateDetector", layout="wide")

# ------------------------------------------------------------------
# Locate results
# ------------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get("AGATEDETECTOR_RESULTS",
                                   Path(__file__).parent / "results"))
GATES_JSON = RESULTS_DIR / "gates.json"

st.title("AutoGateDetector — Results Viewer")

if not GATES_JSON.exists():
    st.warning(f"No results found at {GATES_JSON}. Run pipeline.py first.")
    st.stop()

with open(GATES_JSON) as f:
    data = json.load(f)

# ------------------------------------------------------------------
# Summary header
# ------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cloud", Path(data["cloud_path"]).name)
col2.metric("Gates found", data["n_gates"])
col3.metric("Slices scanned", data["n_slices"])
col4.metric("Elapsed", f"{data['elapsed_s']:.0f}s")

st.caption(
    f"Axis: {data['axis']}  |  Step: {data['step_m']}m  |  "
    f"Thickness: {data['thickness_m']}m  |  "
    f"Z: {data['zmin']}–{data['zmax']}m"
)

if not data["gates"]:
    st.info("No gates detected.")
    st.stop()

# ------------------------------------------------------------------
# Build DataFrame
# ------------------------------------------------------------------

rows = []
for g in data["gates"]:
    bbox = g["bbox_2d"]
    rows.append({
        "gate_id": g["gate_id"],
        "axis": g["axis"],
        "position_m": round(g["position_m"], 2),
        "confidence": round(g["confidence"], 2),
        "pipe_count": g["pipe_count"],
        "area_m2": round(g["opening_area_m2"], 2),
        "u0": round(bbox[0], 2),
        "v0": round(bbox[1], 2),
        "u1": round(bbox[2], 2),
        "v1": round(bbox[3], 2),
        "bbox_3d": g["bbox_3d"],
    })

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# Filters sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    min_pipes = st.number_input("Min pipe count", min_value=0, value=0, step=1)
    axes_sel = st.multiselect("Axis", options=["X", "Y"], default=["X", "Y"])

    st.divider()
    if st.button("Reload results"):
        st.cache_data.clear()
        st.rerun()

mask = (
    (df["confidence"] >= min_conf) &
    (df["pipe_count"] >= min_pipes) &
    (df["axis"].isin(axes_sel))
)
df_filt = df[mask].reset_index(drop=True)

st.subheader(f"{len(df_filt)} gates (filtered from {len(df)} total)")

# ------------------------------------------------------------------
# Table
# ------------------------------------------------------------------

display_cols = ["gate_id", "axis", "position_m", "confidence", "pipe_count",
                "area_m2", "u0", "v0", "u1", "v1"]
st.dataframe(
    df_filt[display_cols],
    use_container_width=True,
    height=300,
)

# Download
csv_bytes = df_filt[display_cols].to_csv(index=False).encode()
st.download_button("Download CSV", csv_bytes, "gates_filtered.csv", "text/csv")

# ------------------------------------------------------------------
# 3D scatter of gate centres
# ------------------------------------------------------------------

st.subheader("Gate positions (3D)")

centres = []
for _, row in df_filt.iterrows():
    b = row["bbox_3d"]
    centres.append({
        "x": (b[0] + b[3]) / 2,
        "y": (b[1] + b[4]) / 2,
        "z": (b[2] + b[5]) / 2,
        "conf": row["confidence"],
        "pipes": row["pipe_count"],
        "label": f"{row['gate_id']}<br>pos={row['position_m']}m conf={row['confidence']} pipes={row['pipe_count']}",
    })

if centres:
    cx = [c["x"] for c in centres]
    cy = [c["y"] for c in centres]
    cz = [c["z"] for c in centres]
    colors = [c["conf"] for c in centres]
    labels = [c["label"] for c in centres]

    fig = go.Figure(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers",
        marker=dict(
            size=6,
            color=colors,
            colorscale="RdYlGn",
            cmin=0.5, cmax=1.0,
            colorbar=dict(title="Confidence"),
            opacity=0.85,
        ),
        text=labels,
        hoverinfo="text",
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        scene=dict(
            bgcolor="#1e1e2e",
            xaxis=dict(title="X (m)", color="#aaa"),
            yaxis=dict(title="Y (m)", color="#aaa"),
            zaxis=dict(title="Z (m)", color="#aaa"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Per-position histogram
# ------------------------------------------------------------------

st.subheader("Gates per position")
pos_counts = df_filt.groupby("position_m").size().reset_index(name="count")
fig2 = go.Figure(go.Bar(
    x=pos_counts["position_m"],
    y=pos_counts["count"],
    marker_color="#4fc3f7",
))
fig2.update_layout(
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#1e1e2e",
    font_color="#ccc",
    xaxis=dict(title="Position along axis (m)", color="#aaa"),
    yaxis=dict(title="Gate count", color="#aaa"),
    margin=dict(l=0, r=0, t=20, b=0),
    height=250,
)
st.plotly_chart(fig2, use_container_width=True)
