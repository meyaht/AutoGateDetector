"""Cloud Viewer — inspect a .npy point cloud in the browser.

Launch via CloudViewer.bat (drag and drop a .npy file onto it),
or run directly: streamlit run cloud_viewer.py
"""

import os
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Cloud Viewer", layout="wide")
st.title("Cloud Viewer")

MAX_PTS = 75_000

default_path = os.environ.get("CLOUD_FILE", "")
path_str = st.text_input("NPY file path", value=default_path,
                          placeholder=r"C:\path\to\cloud.npy")

if not path_str:
    st.info("Enter the path to a .npy point cloud file.")
    st.stop()

p = Path(path_str.strip())
if not p.exists():
    st.error(f"File not found: {p}")
    st.stop()

with st.spinner("Loading..."):
    pts = np.load(str(p))

st.caption(f"{len(pts):,} points  |  showing up to {MAX_PTS:,}")

if len(pts) > MAX_PTS:
    idx = np.random.choice(len(pts), MAX_PTS, replace=False)
    pts = pts[idx]

fig = go.Figure(go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode="markers",
    marker=dict(size=1, color=pts[:, 2], colorscale="Viridis", opacity=0.7),
))
fig.update_layout(
    paper_bgcolor="#1e1e2e",
    scene=dict(
        bgcolor="#1e1e2e",
        xaxis=dict(title="X (m)", color="#aaa"),
        yaxis=dict(title="Y (m)", color="#aaa"),
        zaxis=dict(title="Z (m)", color="#aaa"),
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=700,
)
st.plotly_chart(fig, use_container_width=True)

mins = pts[:, :3].min(axis=0)
maxs = pts[:, :3].max(axis=0)
c1, c2, c3 = st.columns(3)
c1.metric("X range", f"{mins[0]:.2f} – {maxs[0]:.2f} m")
c2.metric("Y range", f"{mins[1]:.2f} – {maxs[1]:.2f} m")
c3.metric("Z range", f"{mins[2]:.2f} – {maxs[2]:.2f} m")
