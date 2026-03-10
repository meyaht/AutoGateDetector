"""AutoGateDetector — headless gate detection pipeline.

Usage:
    python pipeline.py <cloud.npy> [options]

    --axis X|Y|both|auto  Slice axis — X, Y, both (scans XZ and YZ), or auto (default: auto=Y)
    --step 0.5      Step between slices in metres (default: 0.5)
    --thickness 0.5 Slab thickness in metres (default: 0.5)
    --zmin 0.0      Min Z to keep (default: 0.0)
    --zmax 2.0      Max Z to keep (default: 2.0)
    --out results/  Output directory (default: results/)
    --launch        Open Streamlit viewer after detection (default: True)
    --no-launch     Skip Streamlit launch
    --no-gd         Skip pushing results to GateDetector and launching it

Imports detect logic from GateDetector so algorithm improvements apply automatically.
"""

import argparse
import json
import sys
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing from GateDetector without installing it as a package
_GATEDETECTOR = Path(__file__).parent.parent / "GateDetector"
if str(_GATEDETECTOR) not in sys.path:
    sys.path.insert(0, str(_GATEDETECTOR))

import numpy as np
from scipy.signal import find_peaks
from gatedetector.detect import detect_gates, detect_pipe_circles, Gate
from gatedetector.slab import extract_slab, cloud_bounds


def detect_and_align(pts: np.ndarray) -> tuple:
    """Detect primary rack direction and rotate cloud to align to nearest cardinal axis.

    Two improvements over naive longest-run:
    1. Z>0.5m filter for detection only — ground returns create long occupied
       runs at many angles and confuse the detector; above-ground points are
       dominated by the rack structure.
    2. Aspect-ratio scoring: score = longest_run / perp_spread (p10–p90).
       The rack is long AND narrow; scattered equipment and ground clutter
       have a lower aspect ratio and will not win.

    Returns (aligned_pts, rot_deg, [cx, cy], scan_axis).
    """
    # Strip ground returns for detection only — full cloud still used for scanning
    detect_pts = pts[pts[:, 2] > 0.5]
    if len(detect_pts) < 1000:
        detect_pts = pts  # fallback for sparse elevated-only sets

    n = min(200_000, len(detect_pts))
    rng = np.random.default_rng(42)
    xy = detect_pts[rng.choice(len(detect_pts), n, replace=False), :2].astype(np.float64)
    c = xy.mean(axis=0)
    xy_c = xy - c

    bin_m = 0.5
    best_deg = 0
    best_score = -1.0

    for deg in range(180):
        rad = np.deg2rad(deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # Main axis projection → longest consecutive run of occupied bins
        proj = xy_c[:, 0] * cos_a + xy_c[:, 1] * sin_a
        lo = proj.min()
        n_bins = max(1, int((proj.max() - lo) / bin_m) + 1)
        bins = np.clip(((proj - lo) / bin_m).astype(np.int32), 0, n_bins - 1)
        occ = np.zeros(n_bins, dtype=bool)
        occ[bins] = True
        padded = np.concatenate([[False], occ, [False]])
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        longest = int((ends - starts).max()) if len(starts) else 0

        # Perpendicular spread (p10–p90) — narrow = good
        perp = xy_c[:, 0] * (-sin_a) + xy_c[:, 1] * cos_a
        perp_spread = max(float(np.percentile(perp, 90) - np.percentile(perp, 10)), 1.0)

        score = longest / perp_spread
        if score > best_score:
            best_score = score
            best_deg = deg

    snap_deg = round(best_deg / 90) * 90
    rot_deg  = float(snap_deg - best_deg)

    print(f"[pipeline] Axis detection: rack direction={best_deg}°, "
          f"snap={snap_deg}°, rotating by {rot_deg:+.1f}°  "
          f"(aspect score={best_score:.2f}, Z>0.5m pts={len(detect_pts):,})", flush=True)

    # Rotate entire cloud around its XY centroid
    r = np.deg2rad(rot_deg)
    rc, rs = float(np.cos(r)), float(np.sin(r))
    cx, cy = float(c[0]), float(c[1])
    aligned = pts.copy()
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    aligned[:, 0] = dx * rc - dy * rs + cx
    aligned[:, 1] = dx * rs + dy * rc + cy

    # After rotation, rack aligns to snap_deg.
    # Scan axis is perpendicular to the rack:
    #   snap=0°  → rack runs along X → scan slices along Y
    #   snap=90° → rack runs along Y → scan slices along X
    scan_axis = "Y" if snap_deg % 180 == 0 else "X"

    return aligned, rot_deg, [cx, cy], scan_axis


def load_cloud(path: str) -> np.ndarray:
    print(f"[pipeline] Loading {path} ...", flush=True)
    t0 = time.time()
    ext = Path(path).suffix.lower()
    if ext == ".e57":
        pts = _load_e57(path)
    else:
        pts = np.load(path)
        if pts.dtype != np.float32:
            pts = pts.astype(np.float32)
    print(f"[pipeline] Loaded {len(pts):,} pts in {time.time()-t0:.1f}s", flush=True)
    return pts


def _load_e57(path: str) -> np.ndarray:
    import pye57
    e57 = pye57.E57(str(path))
    n_scans = e57.scan_count
    chunks = []
    for idx in range(n_scans):
        print(f"[pipeline]   E57 scan {idx+1}/{n_scans} ...", flush=True)
        data = e57.read_scan(idx, ignore_missing_fields=True)
        x = np.asarray(data["cartesianX"], dtype=np.float32)
        y = np.asarray(data["cartesianY"], dtype=np.float32)
        z = np.asarray(data["cartesianZ"], dtype=np.float32)
        chunks.append(np.column_stack([x, y, z]))
    return np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float32)


def _make_plan_thumb_base(
    pts_z: np.ndarray,
    bounds: dict,
    thumb_w: int = 680,
    thumb_h: int = 1400,
) -> dict:
    """Pre-render the XY cloud scatter as a small plan thumbnail.

    Returns a dict with the PIL Image and coordinate transform params so
    _save_slice_image can copy-and-annotate it per gate without re-rendering
    the scatter every time.
    """
    from PIL import Image

    PAD = 40
    x_min, x_max = bounds["xmin"], bounds["xmax"]
    y_min, y_max = bounds["ymin"], bounds["ymax"]
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = min((thumb_w - 2 * PAD) / x_range, (thumb_h - 2 * PAD) / y_range)

    rng = np.random.default_rng(1)
    n = min(150_000, len(pts_z))
    idx = rng.choice(len(pts_z), n, replace=False)
    sub = pts_z[idx]
    x = sub[:, 0].astype(np.float32)
    y = sub[:, 1].astype(np.float32)

    buf = np.full((thumb_h, thumb_w, 3), (18, 18, 28), dtype=np.uint8)
    px_arr = np.clip((PAD + (x - x_min) * scale).astype(np.int32), 0, thumb_w - 1)
    py_arr = np.clip((thumb_h - PAD - (y - y_min) * scale).astype(np.int32), 0, thumb_h - 1)
    buf[py_arr, px_arr] = (55, 100, 145)

    print(f"[pipeline] Plan thumbnail base rendered ({thumb_w}x{thumb_h})", flush=True)
    return {
        "img":   Image.fromarray(buf),
        "scale": scale,
        "x_min": x_min,
        "y_min": y_min,
        "w":     thumb_w,
        "h":     thumb_h,
        "PAD":   PAD,
    }


def _save_slice_image(
    uv: np.ndarray,
    gates: list,
    pipe_circles: list,
    pos: float,
    axis: str,
    u_label: str,
    v_label: str,
    run_dir: Path,
    plan_thumb: dict | None = None,
) -> str:
    """Render a 2D slice with gate boxes and fitted pipe circles, save as PNG.

    If plan_thumb is provided (from _make_plan_thumb_base), the output is a
    combined image: plan thumbnail (left) + cross-section slice (right).

    Uses Pillow only — no matplotlib required.
    Returns the filename (not full path) for embedding in the JSON.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 2400, 1400
    PAD = 80  # pixel padding around data

    fname = f"slice_{axis}_{pos:.2f}m.png"
    fpath = run_dir / fname

    # Subsample for speed
    uv_plot = uv
    if len(uv) > 100_000:
        rng = np.random.default_rng(0)
        uv_plot = uv[rng.choice(len(uv), 100_000, replace=False)]

    # World → pixel transform (flip V so Z increases upward)
    u_min, v_min = uv_plot.min(axis=0)
    u_max, v_max = uv_plot.max(axis=0)
    u_range = max(u_max - u_min, 1e-6)
    v_range = max(v_max - v_min, 1e-6)
    scale = min((W - 2 * PAD) / u_range, (H - 2 * PAD) / v_range)

    def to_px(u, v):
        px = int(PAD + (u - u_min) * scale)
        py = int(H - PAD - (v - v_min) * scale)
        return px, py

    # Point scatter (vectorised)
    buf = np.full((H, W, 3), (26, 26, 26), dtype=np.uint8)
    pts_px_x = np.clip((PAD + (uv_plot[:, 0] - u_min) * scale).astype(np.int32), 0, W - 1)
    pts_px_y = np.clip((H - PAD - (uv_plot[:, 1] - v_min) * scale).astype(np.int32), 0, H - 1)
    buf[pts_px_y, pts_px_x] = (136, 136, 136)
    img = Image.fromarray(buf)
    draw = ImageDraw.Draw(img)

    # Meter grid — faint lines at each whole-metre world coordinate
    GRID_COL = (55, 55, 65)
    u_start = int(np.floor(u_min))
    v_start = int(np.floor(v_min))
    u_end   = int(np.ceil(u_max)) + 1
    v_end   = int(np.ceil(v_max)) + 1
    for u_m in range(u_start, u_end):
        px = int(PAD + (u_m - u_min) * scale)
        if PAD <= px < W - PAD:
            draw.line([(px, PAD), (px, H - PAD)], fill=GRID_COL, width=2)
    for v_m in range(v_start, v_end):
        py = int(H - PAD - (v_m - v_min) * scale)
        if PAD <= py < H - PAD:
            draw.line([(PAD, py), (W - PAD, py)], fill=GRID_COL, width=2)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 26)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Fitted pipe circles — cyan outlines with nominal size label
    for pc in pipe_circles:
        cx_px, cy_px = to_px(pc["u_m"], pc["v_m"])
        r_px = int(pc["radius_m"] * scale)
        if r_px < 1:
            continue
        draw.ellipse(
            [cx_px - r_px, cy_px - r_px, cx_px + r_px, cy_px + r_px],
            outline=(0, 200, 220), width=2,
        )
        lbl = f"{pc['nominal_in']}\""
        draw.text((cx_px - r_px, cy_px - r_px - 26), lbl, fill=(0, 200, 220), font=font_sm)

    # Gate bounding boxes
    palette = [
        (243, 156, 18), (231, 76, 60), (46, 204, 113),
        (52, 152, 219), (155, 89, 182),
    ]
    for ci, g in enumerate(gates):
        d = g.to_dict()
        u0, v0, u1, v1 = d["bbox_2d"]
        color = palette[ci % len(palette)]
        x0, y0 = to_px(u0, v0)
        x1, y1 = to_px(u1, v1)
        draw.rectangle([x0, min(y0, y1), x1, max(y0, y1)], outline=color, width=4)
        lbl = f"{d['gate_id']}  conf={d['confidence']:.2f}  {d['pipe_count']} pipes"
        draw.text((x0 + 6, min(y0, y1) - 32), lbl, fill=color, font=font)

    # Axis labels
    draw.text((W // 2 - 80, H - 36), f"{u_label} (m)", fill=(180, 180, 180), font=font)
    draw.text((8, H // 2 - 60), f"{v_label}\n(m)", fill=(180, 180, 180), font=font)
    title = (f"Slice {axis}={pos:.2f} m  —  {len(gates)} gate(s)"
             f"  {len(pipe_circles)} pipe(s)")
    draw.text((PAD, 12), title, fill=(255, 255, 255), font=font)

    # --- Plan thumbnail panel (left side) ---
    if plan_thumb is not None:
        tw = plan_thumb["w"]
        th = plan_thumb["h"]
        tpad = plan_thumb["PAD"]
        tscale = plan_thumb["scale"]
        tx_min = plan_thumb["x_min"]
        ty_min = plan_thumb["y_min"]

        thumb = plan_thumb["img"].copy()
        tdraw = ImageDraw.Draw(thumb)

        def to_tpx(xw, yw):
            return (int(tpad + (xw - tx_min) * tscale),
                    int(th - tpad - (yw - ty_min) * tscale))

        # Scan-plane position line
        if axis == "Y":
            _, py_line = to_tpx(tx_min, pos)
            if 0 <= py_line < th:
                tdraw.line([(0, py_line), (tw, py_line)], fill=(200, 200, 60), width=2)
        else:  # X
            px_line, _ = to_tpx(pos, ty_min)
            if 0 <= px_line < tw:
                tdraw.line([(px_line, 0), (px_line, th)], fill=(200, 200, 60), width=2)

        # Gate boxes on thumbnail
        COL_Y_T = (243, 156, 18)
        COL_X_T = (0, 210, 230)
        MIN_TPX = 6
        for g in gates:
            d = g.to_dict()
            b3 = d.get("bbox_3d", [])
            if len(b3) != 6:
                continue
            gx0, gy0, _, gx1, gy1, _ = b3
            ax = d.get("axis", "Y")
            col = COL_Y_T if ax == "Y" else COL_X_T
            tx0, ty0 = to_tpx(gx0, gy0)
            tx1, ty1 = to_tpx(gx1, gy1)
            lx0, lx1 = min(tx0, tx1), max(tx0, tx1)
            ly0, ly1 = min(ty0, ty1), max(ty0, ty1)
            if ax == "Y" and (ly1 - ly0) < MIN_TPX:
                mid = (ly0 + ly1) // 2
                ly0, ly1 = mid - MIN_TPX // 2, mid + MIN_TPX // 2
            if ax == "X" and (lx1 - lx0) < MIN_TPX:
                mid = (lx0 + lx1) // 2
                lx0, lx1 = mid - MIN_TPX // 2, mid + MIN_TPX // 2
            tdraw.rectangle([lx0, ly0, lx1, ly1], outline=col, width=4)

        # Separator + label
        tdraw.line([(tw - 1, 0), (tw - 1, th)], fill=(60, 60, 80), width=4)
        try:
            lbl_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 20)
        except Exception:
            lbl_font = ImageFont.load_default()
        tdraw.text((8, 8), "XY Plan", fill=(120, 120, 140), font=lbl_font)

        combined_h = max(th, H)
        combined = Image.new("RGB", (tw + W, combined_h), color=(26, 26, 26))
        combined.paste(thumb, (0, 0))
        combined.paste(img,   (tw, 0))
        img = combined

    img.save(fpath)
    return fname


def _save_plan_image(pts_z: np.ndarray, all_gates: list, run_dir: Path,
                     px_per_m: int = 60) -> str:
    """Render a high-resolution top-down XY plan view PNG.

    Canvas size is derived from the cloud's world extent at px_per_m pixels/metre,
    so the image can be opened full-screen and panned to read gate labels.

    Returns the filename string for embedding in JSON.
    """
    from PIL import Image, ImageDraw, ImageFont

    PAD = 100   # px border around data extents
    fname = "plan.png"
    fpath = run_dir / fname

    # Subsample for rendering — more points = denser scatter at large scale
    rng = np.random.default_rng(0)
    n = min(1_000_000, len(pts_z))
    idx = rng.choice(len(pts_z), n, replace=False)
    sub = pts_z[idx]

    x = sub[:, 0].astype(np.float32)
    y = sub[:, 1].astype(np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # Expand bounds to fully contain every gate bbox
    for g in all_gates:
        b3 = g.get("bbox_3d", [])
        if len(b3) == 6:
            x_min = min(x_min, b3[0], b3[3])
            x_max = max(x_max, b3[0], b3[3])
            y_min = min(y_min, b3[1], b3[4])
            y_max = max(y_max, b3[1], b3[4])

    W = int((x_max - x_min) * px_per_m) + 2 * PAD
    H = int((y_max - y_min) * px_per_m) + 2 * PAD
    print(f"[pipeline] Plan image: {W}x{H} px  ({x_max-x_min:.0f}x{y_max-y_min:.0f} m "
          f"@ {px_per_m}px/m)", flush=True)

    def to_px(xw, yw):
        return (int(PAD + (xw - x_min) * px_per_m),
                int(H - PAD - (yw - y_min) * px_per_m))

    # Vectorised point scatter
    buf = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)
    px_arr = np.clip((PAD + (x - x_min) * px_per_m).astype(np.int32), 0, W - 1)
    py_arr = np.clip((H - PAD - (y - y_min) * px_per_m).astype(np.int32), 0, H - 1)
    buf[py_arr, px_arr] = (65, 125, 175)

    img = Image.fromarray(buf)
    draw = ImageDraw.Draw(img)

    # Fonts — two sizes
    try:
        font_lbl = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
        font_ax  = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 11)
        font_hdr = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 16)
    except Exception:
        font_lbl = font_ax = font_hdr = ImageFont.load_default()

    # Coordinate grid — lines and tick labels every 10 m
    grid_step = 10
    grid_color = (38, 38, 55)
    tick_color = (90, 90, 110)
    for gx in range(int(x_min // grid_step) * grid_step,
                    int(x_max // grid_step) * grid_step + grid_step + 1,
                    grid_step):
        if x_min - 1 <= gx <= x_max + 1:
            px_g, _ = to_px(gx, y_min)
            draw.line([(px_g, 0), (px_g, H)], fill=grid_color, width=1)
            # Tick label at bottom and top
            lbl = f"{gx}m"
            draw.text((px_g + 3, H - PAD + 4), lbl, fill=tick_color, font=font_ax)
            draw.text((px_g + 3, PAD // 2 - 8), lbl, fill=tick_color, font=font_ax)

    for gy in range(int(y_min // grid_step) * grid_step,
                    int(y_max // grid_step) * grid_step + grid_step + 1,
                    grid_step):
        if y_min - 1 <= gy <= y_max + 1:
            _, py_g = to_px(x_min, gy)
            draw.line([(0, py_g), (W, py_g)], fill=grid_color, width=1)
            lbl = f"{gy}m"
            draw.text((PAD // 2 - 35, py_g - 8), lbl, fill=tick_color, font=font_ax)
            draw.text((W - PAD + 4, py_g - 8), lbl, fill=tick_color, font=font_ax)

    # Gate boxes + labels
    MIN_PX = 6   # minimum box thickness for the slab-depth axis
    COL_Y = (243, 156, 18)   # orange  — Y-axis scan (XZ plane)
    COL_X = (0,  210, 230)   # cyan    — X-axis scan (YZ plane)

    for g in all_gates:
        b3 = g.get("bbox_3d", [])
        if len(b3) != 6:
            continue
        gx0, gy0, _, gx1, gy1, _ = b3
        ax    = g.get("axis", "Y")
        pos   = g.get("position_m", 0.0)
        pipes = g.get("pipe_count", 0)
        gid   = g.get("gate_id", "")
        color = COL_Y if ax == "Y" else COL_X

        x0p, y0p = to_px(gx0, gy0)
        x1p, y1p = to_px(gx1, gy1)
        lx0, lx1 = min(x0p, x1p), max(x0p, x1p)
        ly0, ly1 = min(y0p, y1p), max(y0p, y1p)

        # Enforce minimum visible box thickness on the slab-depth axis
        if ax == "Y" and (ly1 - ly0) < MIN_PX:
            mid = (ly0 + ly1) // 2
            ly0, ly1 = mid - MIN_PX // 2, mid + MIN_PX // 2
        if ax == "X" and (lx1 - lx0) < MIN_PX:
            mid = (lx0 + lx1) // 2
            lx0, lx1 = mid - MIN_PX // 2, mid + MIN_PX // 2

        draw.rectangle([lx0, ly0, lx1, ly1], outline=color, width=2)

        # Label: gate_id on line 1, axis/position/pipes on line 2
        line1 = gid
        line2 = f"{ax}={pos:.1f}m  {pipes}p"
        lbl_x = lx0 + 3
        if ax == "Y":
            # Label above the box
            draw.text((lbl_x, ly0 - 30), line1, fill=color, font=font_lbl)
            draw.text((lbl_x, ly0 - 16), line2, fill=color, font=font_ax)
        else:
            # Label to the right, vertically centered
            cy_box = (ly0 + ly1) // 2
            draw.text((lx1 + 6, cy_box - 15), line1, fill=color, font=font_lbl)
            draw.text((lx1 + 6, cy_box + 1),  line2, fill=color, font=font_ax)

    # Scale bar (50 m)
    bar_m  = 50
    bar_px = bar_m * px_per_m
    bar_x0 = W - PAD - bar_px
    bar_y0 = H - PAD + 50
    draw.line([(bar_x0, bar_y0), (bar_x0 + bar_px, bar_y0)], fill=(200, 200, 200), width=4)
    for tx in [bar_x0, bar_x0 + bar_px]:
        draw.line([(tx, bar_y0 - 8), (tx, bar_y0 + 8)], fill=(200, 200, 200), width=3)
    draw.text((bar_x0 + bar_px // 2 - 20, bar_y0 - 22),
              f"{bar_m} m", fill=(200, 200, 200), font=font_lbl)

    # Title + legend
    y_count = sum(1 for g in all_gates if g.get("axis") == "Y")
    x_count = sum(1 for g in all_gates if g.get("axis") == "X")
    draw.text((PAD, 12),
              f"Plan View (XY)   {y_count} Y-axis gates (orange / XZ plane)"
              f"   {x_count} X-axis gates (cyan / YZ plane)"
              f"   {px_per_m}px/m",
              fill=(210, 210, 210), font=font_hdr)

    img.save(fpath)
    print(f"[pipeline] Saved plan image -> {fpath}  ({W}x{H})", flush=True)
    return fname


def _propagate_pipe_detections(
    all_pipe_detections: list,
    all_gates: list,
    pts_z: np.ndarray,
    step_m: float,
    thickness_m: float,
    run_dir: Path,
    plan_thumb: dict,
) -> list:
    """After the main scan, search for each detected pipe in the adjacent slices.

    A real pipe is a continuous object — if it appears at position P it almost
    certainly appears at P±step too.  For every known circle centre, extract a
    tight window of points from the neighbouring slab and try to fit a circle
    with relaxed thresholds (we have a strong spatial prior).  Any match is
    merged into that slice's record and the image is re-rendered.

    Returns the updated all_pipe_detections list (may contain new entries for
    previously empty adjacent slices).
    """
    if not all_pipe_detections:
        return all_pipe_detections

    print(f"\n[pipeline] ── Pipe propagation pass ──────────────────────────", flush=True)

    _rnd = lambda p: round(float(p), 6)

    # Build lookups keyed by (axis, pos)
    pos_circles  = {}   # (axis, pos) → current circle list
    pos_det_idx  = {}   # (axis, pos) → index in all_pipe_detections
    for i, det in enumerate(all_pipe_detections):
        key = (det["axis"], _rnd(det["position_m"]))
        pos_circles[key] = list(det["circles"])
        pos_det_idx[key] = i

    gate_map = {}       # (axis, pos) → list of gate dicts
    for g in all_gates:
        if g.get("gate_id") == "_CLOUD_META_":
            continue
        key = (g.get("axis", "Y"), _rnd(g.get("position_m", 0)))
        gate_map.setdefault(key, []).append(g)

    # ── Search pass ─────────────────────────────────────────────────────────
    # Cache slab extractions so each adjacent position is only extracted once.
    uv_cache  = {}   # (axis, adj_pos) → (uv, u_label, v_label) | None
    additions = {}   # (axis, adj_pos) → list of newly-found circles

    n_total = len(all_pipe_detections)
    print(f"[pipeline] Propagation: checking {n_total} source slice(s)…", flush=True)
    t_prop  = time.time()
    BAR_W   = 28

    def _bar(done, total, new_found, elapsed):
        frac   = done / max(total, 1)
        filled = int(BAR_W * frac)
        bar    = "=" * filled + (">" if filled < BAR_W else "=") + " " * max(0, BAR_W - filled - 1)
        return (f"\r  [{bar}] {done}/{total}  "
                f"new={new_found}  {elapsed:.0f}s  ")

    for det_i, det in enumerate(all_pipe_detections):
        axis  = det["axis"]
        pos   = det["position_m"]
        known = det["circles"]
        sys.stdout.write(_bar(det_i + 1, n_total,
                              sum(len(v) for v in additions.values()),
                              time.time() - t_prop))
        sys.stdout.flush()
        if not known:
            continue

        for direction in (-1, 1):
            adj_pos = _rnd(pos + direction * step_m)
            adj_key = (axis, adj_pos)

            if adj_key not in uv_cache:
                _, uv_adj, u_lbl, v_lbl = extract_slab(
                    pts_z, axis=axis, position_m=adj_pos, thickness_m=thickness_m,
                )
                uv_cache[adj_key] = (
                    (uv_adj, u_lbl, v_lbl)
                    if (uv_adj is not None and len(uv_adj) >= 10)
                    else None
                )

            cached = uv_cache[adj_key]
            if cached is None:
                continue
            uv_adj = cached[0]

            existing = pos_circles.get(adj_key, [])
            pending  = additions.get(adj_key, [])

            for pc in known:
                u_c, v_c, r_c = pc["u_m"], pc["v_m"], pc["radius_m"]

                # Skip if a circle is already close to this centre in the adj slice
                if any(
                    abs(e["u_m"] - u_c) < 0.15 and abs(e["v_m"] - v_c) < 0.15
                    for e in existing + pending
                ):
                    continue

                # Extract a tight window around the expected circle centre
                sr = r_c + 0.20   # radius + 200 mm margin
                mask = (
                    (uv_adj[:, 0] >= u_c - sr) & (uv_adj[:, 0] <= u_c + sr) &
                    (uv_adj[:, 1] >= v_c - sr) & (uv_adj[:, 1] <= v_c + sr)
                )
                nearby = uv_adj[mask]
                if len(nearby) < 6:
                    continue

                # Relax thresholds — spatial prior justifies a lower bar
                candidates = detect_pipe_circles(
                    nearby, min_inlier_frac=0.40, min_arc_deg=120.0,
                )
                for fc in candidates:
                    if abs(fc["u_m"] - u_c) < 0.15 and abs(fc["v_m"] - v_c) < 0.15:
                        additions.setdefault(adj_key, []).append(fc)
                        break   # one match per known pipe

    sys.stdout.write("\n")
    sys.stdout.flush()

    if not additions:
        print(f"[pipeline] Propagation: no new circles found.", flush=True)
        return all_pipe_detections

    total_new = sum(len(v) for v in additions.values())
    print(f"[pipeline] Propagation: {total_new} new pipe circle(s) across "
          f"{len(additions)} slice(s).", flush=True)

    # ── Apply pass: merge and re-render ─────────────────────────────────────
    updated = list(all_pipe_detections)

    for adj_key, new_circles in sorted(additions.items()):
        axis, adj_pos = adj_key
        cached = uv_cache.get(adj_key)
        if cached is None:
            continue
        uv_adj, u_lbl, v_lbl = cached

        _GATE_EXTRA = {"slice_image"}
        gate_objs  = [Gate.from_dict({k: v for k, v in g.items() if k not in _GATE_EXTRA})
                      for g in gate_map.get(adj_key, [])]
        all_circles = pos_circles.get(adj_key, []) + new_circles

        img_fname = _save_slice_image(
            uv_adj, gate_objs, all_circles, adj_pos,
            axis, u_lbl, v_lbl, run_dir, plan_thumb=plan_thumb,
        )
        print(f"[pipeline]   {axis}={adj_pos:.2f}m  +{len(new_circles)} propagated pipe(s)"
              f"  → {img_fname}", flush=True)

        if adj_key in pos_det_idx:
            i = pos_det_idx[adj_key]
            rec = dict(updated[i])
            rec["circles"]     = all_circles
            rec["slice_image"] = img_fname
            updated[i] = rec
        else:
            updated.append({
                "axis":        axis,
                "position_m":  adj_pos,
                "slice_image": img_fname,
                "circles":     all_circles,
            })

    return updated


def _has_full_width_hband(uv: np.ndarray,
                          min_span_frac: float = 0.60,
                          strip_h_m: float = 0.090) -> bool:
    """Return True if any horizontal strip of points spans >= min_span_frac
    of the total U width — indicating a structural beam chord crossing the slice.
    Strip height is 3 × 30 mm cells by default.
    """
    if len(uv) < 20:
        return False
    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)
    u_range = u_max - u_min
    if u_range < 1.0:
        return False
    n_strips = max(1, int((v_max - v_min) / strip_h_m))
    v_edges  = np.linspace(v_min, v_max, n_strips + 1)
    for i in range(n_strips):
        mask = (uv[:, 1] >= v_edges[i]) & (uv[:, 1] < v_edges[i + 1])
        if mask.sum() < 5:
            continue
        span = uv[mask, 0].max() - uv[mask, 0].min()
        if span >= u_range * min_span_frac:
            return True
    return False


def _has_tall_vband(uv: np.ndarray,
                    min_height_m: float = 1.0,
                    strip_w_m: float = 0.090) -> bool:
    """Return True if any narrow vertical column of points spans >= min_height_m
    — indicating a structural column crossing the slice.
    Strip width is 3 × 30 mm cells by default.
    """
    if len(uv) < 10:
        return False
    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)
    n_strips = max(1, int((u_max - u_min) / strip_w_m))
    u_edges  = np.linspace(u_min, u_max, n_strips + 1)
    for i in range(n_strips):
        mask = (uv[:, 0] >= u_edges[i]) & (uv[:, 0] < u_edges[i + 1])
        if mask.sum() < 5:
            continue
        height = uv[mask, 1].max() - uv[mask, 1].min()
        if height >= min_height_m:
            return True
    return False


def _struct_pass(
    pts_z: np.ndarray,
    scan_axis: str,
    bounds: dict,
    struct_thickness_m: float = 0.05,
    struct_step_m: float = 0.05,
    min_h_span_frac: float = 0.60,
    min_col_height_m: float = 1.0,
    cluster_gap_m: float = 0.50,
) -> list[float]:
    """Thin-slice structural pre-pass: locate frame bent positions.

    Scans at fine resolution (default 50 mm step, 50 mm thick) looking for
    slices that contain BOTH:
      • a full-width horizontal band  → beam chord spanning the rack
      • a tall vertical band          → structural column

    Cross-bracing and pipes are excluded: bracing spans only a tiny horizontal
    fraction; pipes are thin rings that don't span the full width.

    Nearby hits within cluster_gap_m are merged to one representative position
    (the median of the cluster).  Returns a sorted list of bent positions.
    """
    ax_min = bounds[{"X": "xmin", "Y": "ymin"}[scan_axis]]
    ax_max = bounds[{"X": "xmax", "Y": "ymax"}[scan_axis]]
    positions = np.arange(ax_min + struct_step_m / 2, ax_max, struct_step_m)

    n_pos     = len(positions)
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    BAR_W     = 28
    print(f"[pipeline] ─── {scan_axis}-struct pre-pass  "
          f"({n_pos} thin slices, {ax_min:.1f}–{ax_max:.1f} m, "
          f"thickness={struct_thickness_m}m, workers={n_workers}) ───", flush=True)

    hits     = []
    done_ctr = [0]
    bar_lock = threading.Lock()

    def _check_pos(pos):
        _, uv, _, _ = extract_slab(
            pts_z, axis=scan_axis, position_m=float(pos),
            thickness_m=struct_thickness_m,
        )
        if uv is not None and len(uv) >= 20:
            if (_has_full_width_hband(uv, min_span_frac=min_h_span_frac) and
                    _has_tall_vband(uv, min_height_m=min_col_height_m)):
                return float(pos)
        return None

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_check_pos, pos): pos for pos in positions}
        for fut in as_completed(futures):
            result = fut.result()
            with bar_lock:
                done_ctr[0] += 1
                if result is not None:
                    hits.append(result)
                frac   = done_ctr[0] / max(n_pos, 1)
                filled = int(BAR_W * frac)
                bar    = "=" * filled + (">" if filled < BAR_W else "=") + " " * max(0, BAR_W - filled - 1)
                sys.stdout.write(f"\r  {scan_axis}-struct: [{bar}] {done_ctr[0]}/{n_pos}  "
                                 f"frames={len(hits)}  ")
                sys.stdout.flush()

    hits.sort()
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not hits:
        print(f"[pipeline] ─── {scan_axis}-struct: no frames found ───", flush=True)
        return []

    # Cluster nearby hits → one position per bent
    clusters: list[list[float]] = []
    group: list[float] = [hits[0]]
    for p in hits[1:]:
        if p - group[-1] <= cluster_gap_m:
            group.append(p)
        else:
            clusters.append(group)
            group = [p]
    clusters.append(group)

    bent_positions = [float(np.median(c)) for c in clusters]
    print(f"[pipeline] ─── {scan_axis}-struct: {len(bent_positions)} frame bent(s) located ───",
          flush=True)
    return sorted(bent_positions)


def _has_structural_support(uv: np.ndarray, pc: dict,
                            look_below_m: float = 0.15,
                            min_pts: int = 10) -> bool:
    """Return True if there are enough points in a horizontal band just below
    the pipe bottom, indicating a structural support (beam / adjacent pipe row).

    The beam sits at the lower quadrant of the pipe or just below it — focus
    the search window tightly around the pipe bottom (default 150 mm window)
    rather than a broad 300 mm zone that may catch unrelated structure.

    Pipes in a rack always rest on something.  A circle whose floor zone is
    empty is almost certainly a false positive floating between beam rows.
    """
    u_c, v_c, r_c = pc["u_m"], pc["v_m"], pc["radius_m"]
    mask = (
        (uv[:, 0] >= u_c - r_c - 0.10) & (uv[:, 0] <= u_c + r_c + 0.10) &
        (uv[:, 1] >= v_c - r_c - look_below_m) & (uv[:, 1] <= v_c - r_c + 0.05)
    )
    return int(mask.sum()) >= min_pts


def _bottom_beam_pipe_search(
    uv: np.ndarray,
    gates: list,
    existing_circles: list,
) -> list:
    """Targeted pipe search along the bottom beam of each gate.

    Pipes resting on the bottom chord have their lower arc buried against the
    beam; only the upper ~150° is visible.  When many pipes are packed side by
    side the standard connected-component fit merges them into one blob.

    Strategy:
      1. Restrict to a vertical band just above the gate bottom (v0 → v0+0.6 m).
      2. Build a 1D histogram along the horizontal axis to locate individual pipe
         columns as peaks — each peak is a candidate pipe centre.
      3. For each peak, extract a narrow column of points and attempt a circle
         fit with relaxed thresholds (arc ≥ 130°).
    """
    new_circles = []
    already = {(round(pc["u_m"], 2), round(pc["v_m"], 2)) for pc in existing_circles}

    for g in gates:
        u0, v0, u1, v1 = g.to_dict()["bbox_2d"]

        # Bottom-beam zone: 50 mm below v0 to 600 mm above it
        bz_mask = (
            (uv[:, 0] >= u0 - 0.10) & (uv[:, 0] <= u1 + 0.10) &
            (uv[:, 1] >= v0 - 0.05) & (uv[:, 1] <= v0 + 0.60)
        )
        uv_zone = uv[bz_mask]
        if len(uv_zone) < 20:
            continue

        # 1D histogram along horizontal axis — 20 mm bins
        bin_w = 0.020
        bins = np.arange(u0 - 0.10, u1 + 0.10 + bin_w, bin_w)
        hist, edges = np.histogram(uv_zone[:, 0], bins=bins)

        # Peaks separated by at least 60 mm (smallest pipe OD = 60 mm)
        min_sep = max(1, int(0.060 / bin_w))
        peaks, _ = find_peaks(hist, distance=min_sep, height=4)

        for pi in peaks:
            u_peak = float((edges[pi] + edges[pi + 1]) / 2)

            # Skip if already have a circle near this horizontal position
            if any(abs(u_peak - pc["u_m"]) < 0.10
                   for pc in existing_circles + new_circles):
                continue

            # Extract narrow column around peak, tall enough for any pipe size
            col_mask = (
                (uv[:, 0] >= u_peak - 0.25) & (uv[:, 0] <= u_peak + 0.25) &
                (uv[:, 1] >= v0 - 0.05)     & (uv[:, 1] <= v0 + 0.55)
            )
            nearby = uv[col_mask]
            if len(nearby) < 6:
                continue

            # Relaxed arc: upper half only (pipe resting on beam)
            candidates = detect_pipe_circles(
                nearby, min_inlier_frac=0.40, min_arc_deg=130.0,
            )
            for c in candidates:
                key = (round(c["u_m"], 2), round(c["v_m"], 2))
                if key not in already and u0 - 0.15 <= c["u_m"] <= u1 + 0.15:
                    if _has_structural_support(uv, c):
                        new_circles.append(c)
                        already.add(key)
                        break   # one pipe per peak

    return new_circles


def run_pipeline(
    cloud_path: str,
    axis: str = "auto",
    step_m: float = 0.5,
    thickness_m: float = 0.5,
    zmin: float = 0.0,
    zmax: float = 2.0,
    out_dir: str = "results",
    launch: bool = True,
    launch_gd: bool = True,
) -> Path:
    pts = load_cloud(cloud_path)

    # Z filter
    z_col = 2
    mask = (pts[:, z_col] >= zmin) & (pts[:, z_col] <= zmax)
    pts_z = pts[mask]
    print(f"[pipeline] After Z filter ({zmin}–{zmax}m): {len(pts_z):,} pts", flush=True)

    rot_deg    = 0.0
    align_center = [float(pts_z[:, 0].mean()), float(pts_z[:, 1].mean())]
    # Auto-alignment disabled — align cloud manually in CloudCompare before running
    # if axis.lower() == "auto":
    #     pts_z, rot_deg, align_center, axis = detect_and_align(pts_z)
    if axis.lower() == "auto":
        axis = "Y"  # default scan axis when alignment is disabled

    scan_axes = ["X", "Y"] if axis.upper() == "BOTH" else [axis.upper()]

    bounds = cloud_bounds(pts_z)

    # Create timestamped run directory up front so images go there immediately
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-render plan thumbnail once (reused by every slice image)
    plan_thumb = _make_plan_thumb_base(pts_z, bounds)

    all_gates = []
    all_pipe_detections = []  # per-slice pipe circle records
    n_slices_total = 0
    t_start = time.time()

    for scan_axis in scan_axes:
        ax_key_min = {"X": "xmin", "Y": "ymin"}[scan_axis]
        ax_key_max = {"X": "xmax", "Y": "ymax"}[scan_axis]
        ax_min = bounds[ax_key_min]
        ax_max = bounds[ax_key_max]

        # Structural pre-pass: locate frame bents, then only scan those positions
        struct_positions = _struct_pass(pts_z, scan_axis, bounds)
        if struct_positions:
            positions = np.array(struct_positions)
        else:
            # Fallback: no frames found — scan everything as before
            print(f"[pipeline] WARNING: struct pass found no frames on {scan_axis}, "
                  f"falling back to full sweep.", flush=True)
            positions = np.arange(ax_min + step_m / 2, ax_max, step_m)

        n_slices_total += len(positions)
        n_pos = len(positions)
        print(f"", flush=True)
        print(f"[pipeline] ═══ {scan_axis}-axis scan  ({n_pos} slices at frame positions, "
              f"{ax_min:.1f}–{ax_max:.1f} m) ═══", flush=True)
        axis_gate_count  = 0
        axis_pipe_count  = 0
        BAR_W = 28
        bar_lock = threading.Lock()
        done_count = [0]
        axis_gate_count_arr = [0]
        axis_pipe_count_arr = [0]

        def _scan_bar(done, gates_n, pipes_n, elapsed):
            frac   = done / max(n_pos, 1)
            filled = int(BAR_W * frac)
            bar    = "=" * filled + (">" if filled < BAR_W else "=") + " " * max(0, BAR_W - filled - 1)
            return (f"\r  {scan_axis}: [{bar}] {done}/{n_pos}  "
                    f"gates={gates_n}  pipes={pipes_n}  {elapsed:.0f}s  ")

        def _scan_worker(pos):
            _, uv, u_label, v_label = extract_slab(
                pts_z, axis=scan_axis, position_m=float(pos), thickness_m=thickness_m,
            )
            if uv is None or len(uv) < 10:
                return {"pos": float(pos), "gates": [], "pipe_circles": [],
                        "img_fname": None, "u_label": u_label, "v_label": v_label}

            gates, _ = detect_gates(
                uv, axis=scan_axis, position_m=float(pos), thickness_m=thickness_m,
                pts3d=pts_z, verbose=False,
            )
            pipe_circles = detect_pipe_circles(uv)

            if gates and pipe_circles:
                gate_bboxes = [g.to_dict()["bbox_2d"] for g in gates]
                margin = 0.15
                def _in_any_gate(pc, _bboxes=gate_bboxes, _m=margin):
                    u, v = pc["u_m"], pc["v_m"]
                    for u0, v0, u1, v1 in _bboxes:
                        if u0 - _m <= u <= u1 + _m and v0 - _m <= v <= v1 + _m:
                            return True
                    return False
                pipe_circles = [pc for pc in pipe_circles if _in_any_gate(pc)]

            if pipe_circles:
                pipe_circles = [pc for pc in pipe_circles
                                if _has_structural_support(uv, pc)]

            if gates:
                bottom_extras = _bottom_beam_pipe_search(uv, gates, pipe_circles)
                pipe_circles = pipe_circles + bottom_extras

            img_fname = None
            if gates or pipe_circles:
                img_fname = _save_slice_image(uv, gates, pipe_circles, float(pos),
                                              scan_axis, u_label, v_label, run_dir,
                                              plan_thumb=plan_thumb)

            return {"pos": float(pos), "gates": gates, "pipe_circles": pipe_circles,
                    "img_fname": img_fname, "u_label": u_label, "v_label": v_label}

        n_workers = max(1, (os.cpu_count() or 4) - 1)
        scan_results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {executor.submit(_scan_worker, pos): pos for pos in positions}
            for future in as_completed(future_map):
                result = future.result()
                scan_results[result["pos"]] = result
                with bar_lock:
                    done_count[0] += 1
                    if result["gates"]:
                        axis_gate_count_arr[0] += len(result["gates"])
                        sys.stdout.write(
                            f"\n[pipeline]   {scan_axis}={result['pos']:.1f}m → "
                            f"{len(result['gates'])} gates, {len(result['pipe_circles'])} pipe(s)\n"
                        )
                    axis_pipe_count_arr[0] += len(result["pipe_circles"])
                    sys.stdout.write(_scan_bar(done_count[0], axis_gate_count_arr[0],
                                               axis_pipe_count_arr[0], time.time() - t_start))
                    sys.stdout.flush()

        axis_gate_count = axis_gate_count_arr[0]
        axis_pipe_count = axis_pipe_count_arr[0]

        # Merge results in position order
        for pos in sorted(scan_results.keys()):
            r = scan_results[pos]
            if r["gates"] or r["pipe_circles"]:
                for g in r["gates"]:
                    d = g.to_dict()
                    d["slice_image"] = r["img_fname"]
                    all_gates.append(d)
                if r["pipe_circles"]:
                    all_pipe_detections.append({
                        "axis":        scan_axis,
                        "position_m":  pos,
                        "slice_image": r["img_fname"],
                        "circles":     r["pipe_circles"],
                    })

        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[pipeline] ═══ {scan_axis}-axis done — {axis_gate_count} gates found ═══", flush=True)

    elapsed = time.time() - t_start
    print(f"[pipeline] Done — {len(all_gates)} gates total in {elapsed:.0f}s", flush=True)

    # Propagation pass: find missed pipes in adjacent slices
    all_pipe_detections = _propagate_pipe_detections(
        all_pipe_detections, all_gates, pts_z, step_m, thickness_m, run_dir, plan_thumb,
    )

    # Generate plan view image
    plan_fname = _save_plan_image(pts_z, all_gates, run_dir)

    out_path = run_dir

    cloud_bmin = [float(bounds["xmin"]), float(bounds["ymin"]), float(bounds["zmin"])]
    cloud_bmax = [float(bounds["xmax"]), float(bounds["ymax"]), float(bounds["zmax"])]

    # Prepend _CLOUD_META_ so GateDetector can restore display context
    meta_entry = {
        "gate_id":      "_CLOUD_META_",
        "plan_rotation": round(rot_deg, 2),
        "align_center":  align_center,
        "cloud_bmin":    cloud_bmin,
        "cloud_bmax":    cloud_bmax,
        "plan_image":    plan_fname,
    }
    gates_with_meta = [meta_entry] + all_gates

    result = {
        "cloud_path":      cloud_path,
        "axis":            "BOTH" if len(scan_axes) > 1 else scan_axes[0],
        "align_angle_deg": round(rot_deg, 2),
        "step_m":          step_m,
        "thickness_m":     thickness_m,
        "zmin":            zmin,
        "zmax":            zmax,
        "n_slices":        n_slices_total,
        "n_gates":         len(all_gates),
        "n_pipe_slices":   len(all_pipe_detections),
        "elapsed_s":       round(elapsed, 1),
        "gates":           gates_with_meta,
        "pipe_detections": all_pipe_detections,
    }

    gates_json = out_path / f"{ts}_gates.json"
    with open(gates_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[pipeline] Saved → {gates_json}", flush=True)

    # Also save CSV for quick inspection
    import csv
    gates_csv = out_path / f"{ts}_gates.csv"
    if all_gates:
        cols = ["gate_id", "axis", "position_m", "confidence", "pipe_count",
                "opening_area_m2", "bbox_2d", "slice_image"]
        with open(gates_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for g in all_gates:
                row = {k: g[k] for k in cols}
                row["bbox_2d"] = str(g["bbox_2d"])
                w.writerow(row)
        print(f"[pipeline] Saved → {gates_csv}", flush=True)

    if launch:
        _launch_viewer(run_dir)

    if launch_gd:
        _push_and_launch_gatedetector(gates_json)

    return gates_json


def _launch_viewer(out_dir: Path):
    viewer = Path(__file__).parent / "app.py"
    env = os.environ.copy()
    env["AGATEDETECTOR_RESULTS"] = str(out_dir)
    print(f"[pipeline] Launching Streamlit viewer ...", flush=True)
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(viewer),
         "--server.port", "8053", "--server.headless", "false"],
        env=env,
        cwd=str(Path(__file__).parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _push_and_launch_gatedetector(gates_json: Path):
    gd_dir = _GATEDETECTOR
    print(f"[pipeline] Launching GateDetector with {gates_json.name} ...", flush=True)
    subprocess.call(
        [sys.executable, str(gd_dir / "app.py"), str(gates_json)],
        cwd=str(gd_dir),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGateDetector — headless gate detection")
    parser.add_argument("cloud", help="Path to .npy point cloud")
    parser.add_argument("--axis", default="auto", choices=["X", "Y", "both", "auto"],
                        help="Scan axis — X, Y, both, or auto (default: auto)")
    parser.add_argument("--step", type=float, default=0.5,
                        help="Slice step in metres (default: 0.5)")
    parser.add_argument("--thickness", type=float, default=0.5,
                        help="Slab thickness in metres (default: 0.5)")
    parser.add_argument("--zmin", type=float, default=0.0,
                        help="Min Z to keep (default: 0.0)")
    parser.add_argument("--zmax", type=float, default=2.0,
                        help="Max Z to keep (default: 2.0)")
    parser.add_argument("--out", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--no-launch", action="store_true",
                        help="Skip Streamlit launch")
    parser.add_argument("--no-gd", action="store_true",
                        help="Skip pushing results to GateDetector")
    args = parser.parse_args()

    # Resolve out dir relative to this script
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir

    run_pipeline(
        cloud_path=args.cloud,
        axis=args.axis,
        step_m=args.step,
        thickness_m=args.thickness,
        zmin=args.zmin,
        zmax=args.zmax,
        out_dir=str(out_dir),
        launch=not args.no_launch,
        launch_gd=not args.no_gd,
    )
