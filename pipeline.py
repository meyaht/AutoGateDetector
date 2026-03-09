"""AutoGateDetector — headless gate detection pipeline.

Usage:
    python pipeline.py <cloud.npy> [options]

    --axis X|Y|both|auto  Slice axis — X, Y, both (scans XZ and YZ), or auto (default: auto=Y)
    --step 0.5      Step between slices in metres (default: 0.5)
    --thickness 0.5 Slab thickness in metres (default: 0.5)
    --zmin 0.0      Min Z to keep (default: 0.0)
    --zmax 7.0      Max Z to keep (default: 7.0)
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
import time
from pathlib import Path

# Allow importing from GateDetector without installing it as a package
_GATEDETECTOR = Path(__file__).parent.parent / "GateDetector"
if str(_GATEDETECTOR) not in sys.path:
    sys.path.insert(0, str(_GATEDETECTOR))

import numpy as np
from gatedetector.detect import detect_gates, detect_pipe_circles
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
    thumb_w: int = 340,
    thumb_h: int = 700,
) -> dict:
    """Pre-render the XY cloud scatter as a small plan thumbnail.

    Returns a dict with the PIL Image and coordinate transform params so
    _save_slice_image can copy-and-annotate it per gate without re-rendering
    the scatter every time.
    """
    from PIL import Image

    PAD = 20
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

    W, H = 1200, 700
    PAD = 40  # pixel padding around data

    fname = f"slice_{axis}_{pos:.2f}m.png"
    fpath = run_dir / fname

    # Subsample for speed
    uv_plot = uv
    if len(uv) > 50_000:
        rng = np.random.default_rng(0)
        uv_plot = uv[rng.choice(len(uv), 50_000, replace=False)]

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

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 11)
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
            outline=(0, 200, 220), width=1,
        )
        lbl = f"{pc['nominal_in']}\""
        draw.text((cx_px - r_px, cy_px - r_px - 13), lbl, fill=(0, 200, 220), font=font_sm)

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
        draw.rectangle([x0, min(y0, y1), x1, max(y0, y1)], outline=color, width=2)
        lbl = f"{d['gate_id']}  conf={d['confidence']:.2f}  {d['pipe_count']} pipes"
        draw.text((x0 + 3, min(y0, y1) - 16), lbl, fill=color, font=font)

    # Axis labels
    draw.text((W // 2 - 40, H - 18), f"{u_label} (m)", fill=(180, 180, 180), font=font)
    draw.text((4, H // 2 - 30), f"{v_label}\n(m)", fill=(180, 180, 180), font=font)
    title = (f"Slice {axis}={pos:.2f} m  —  {len(gates)} gate(s)"
             f"  {len(pipe_circles)} pipe(s)")
    draw.text((PAD, 6), title, fill=(255, 255, 255), font=font)

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
                tdraw.line([(0, py_line), (tw, py_line)], fill=(200, 200, 60), width=1)
        else:  # X
            px_line, _ = to_tpx(pos, ty_min)
            if 0 <= px_line < tw:
                tdraw.line([(px_line, 0), (px_line, th)], fill=(200, 200, 60), width=1)

        # Gate boxes on thumbnail
        COL_Y_T = (243, 156, 18)
        COL_X_T = (0, 210, 230)
        MIN_TPX = 3
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
            tdraw.rectangle([lx0, ly0, lx1, ly1], outline=col, width=2)

        # Separator + label
        tdraw.line([(tw - 1, 0), (tw - 1, th)], fill=(60, 60, 80), width=2)
        try:
            lbl_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 10)
        except Exception:
            lbl_font = ImageFont.load_default()
        tdraw.text((4, 4), "XY Plan", fill=(120, 120, 140), font=lbl_font)

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


def run_pipeline(
    cloud_path: str,
    axis: str = "auto",
    step_m: float = 0.5,
    thickness_m: float = 0.5,
    zmin: float = 0.0,
    zmax: float = 7.0,
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

        positions = np.arange(ax_min + step_m / 2, ax_max, step_m)
        n_slices_total += len(positions)
        print(f"", flush=True)
        print(f"[pipeline] ═══ {scan_axis}-axis scan  ({len(positions)} slices, "
              f"{ax_min:.1f}–{ax_max:.1f} m, step={step_m}m) ═══", flush=True)
        axis_gate_count = 0

        for i, pos in enumerate(positions):
            _, uv, u_label, v_label = extract_slab(
                pts_z, axis=scan_axis, position_m=float(pos), thickness_m=thickness_m,
            )
            if uv is None or len(uv) < 10:
                continue

            gates, debug_str = detect_gates(
                uv, axis=scan_axis, position_m=float(pos), thickness_m=thickness_m,
                pts3d=pts_z,
            )
            pipe_circles = detect_pipe_circles(uv)

            # Restrict circles to gate bounding boxes (+ 150 mm margin).
            # Outside-gate detections are almost always false positives from
            # structural members, flanges, and partial arcs of beams.
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

            if gates or pipe_circles:
                if gates:
                    axis_gate_count += len(gates)
                    print(f"[pipeline]   {scan_axis}={pos:.1f}m → {len(gates)} gates, "
                          f"{len(pipe_circles)} pipe(s) ({debug_str})", flush=True)
                elif pipe_circles:
                    print(f"[pipeline]   {scan_axis}={pos:.1f}m → {len(pipe_circles)} pipe(s) (no gates)", flush=True)
                img_fname = _save_slice_image(uv, gates, pipe_circles, float(pos),
                                              scan_axis, u_label, v_label, run_dir,
                                              plan_thumb=plan_thumb)
                for g in gates:
                    d = g.to_dict()
                    d["slice_image"] = img_fname
                    all_gates.append(d)
                if pipe_circles:
                    all_pipe_detections.append({
                        "axis":        scan_axis,
                        "position_m":  float(pos),
                        "slice_image": img_fname,
                        "circles":     pipe_circles,
                    })
            elif (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                pct = (i + 1) / len(positions) * 100
                print(f"[pipeline]   {scan_axis} {pct:.0f}% ({i+1}/{len(positions)}) elapsed={elapsed:.0f}s", flush=True)

        print(f"[pipeline] ═══ {scan_axis}-axis done — {axis_gate_count} gates found ═══", flush=True)

    elapsed = time.time() - t_start
    print(f"[pipeline] Done — {len(all_gates)} gates total in {elapsed:.0f}s", flush=True)

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
    parser.add_argument("--zmax", type=float, default=7.0,
                        help="Max Z to keep (default: 7.0)")
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
