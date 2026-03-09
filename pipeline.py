"""AutoGateDetector — headless gate detection pipeline.

Usage:
    python pipeline.py <cloud.npy> [options]

    --axis X|Y|auto Slice axis — auto detects via longest-run (default: auto)
    --step 1.0      Step between slices in metres (default: 1.0)
    --thickness 0.3 Slab thickness in metres (default: 0.3)
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
from gatedetector.detect import detect_gates
from gatedetector.slab import extract_slab, cloud_bounds


def detect_primary_axis(pts: np.ndarray) -> str:
    """Find primary rack axis (X or Y) using longest-run detection.

    Projects XY points at angles 0–179° in 1° steps, bins at 0.5 m, and finds
    the angle with the longest consecutive run of occupied bins. That angle is
    the rack's structural direction. Returns 'X' if the rack runs along X
    (scan slices perpendicular = along Y) or 'Y' if it runs along Y.
    """
    n = min(200_000, len(pts))
    rng = np.random.default_rng(42)
    xy = pts[rng.choice(len(pts), n, replace=False), :2].astype(np.float64)
    c = xy.mean(axis=0)
    xy_c = xy - c

    bin_m = 0.5
    best_deg = 0
    best_run = -1

    for deg in range(180):
        rad = np.deg2rad(deg)
        proj = xy_c[:, 0] * np.cos(rad) + xy_c[:, 1] * np.sin(rad)
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
        if longest > best_run:
            best_run = longest
            best_deg = deg

    # best_deg is the rack's long-axis direction.
    # Scan slices must be perpendicular to the rack → slice axis is 90° from best_deg.
    # If rack runs ~0° or ~180° (along X), slice along Y. If ~90° (along Y), slice along X.
    perp_deg = (best_deg + 90) % 180
    axis = "X" if perp_deg >= 45 and perp_deg < 135 else "Y"
    print(f"[pipeline] Longest-run axis detection: rack direction={best_deg}°, "
          f"scan axis={axis}  (run={best_run} bins × {bin_m}m)", flush=True)
    return axis


def load_cloud(path: str) -> np.ndarray:
    print(f"[pipeline] Loading {path} ...", flush=True)
    t0 = time.time()
    pts = np.load(path)
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)
    print(f"[pipeline] Loaded {len(pts):,} pts in {time.time()-t0:.1f}s", flush=True)
    return pts


def run_pipeline(
    cloud_path: str,
    axis: str = "auto",
    step_m: float = 1.0,
    thickness_m: float = 0.3,
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

    if axis.lower() == "auto":
        axis = detect_primary_axis(pts_z)

    bounds = cloud_bounds(pts_z)
    ax_key_min = {"X": "xmin", "Y": "ymin"}[axis.upper()]
    ax_key_max = {"X": "xmax", "Y": "ymax"}[axis.upper()]
    ax_min = bounds[ax_key_min]
    ax_max = bounds[ax_key_max]

    positions = np.arange(ax_min + step_m / 2, ax_max, step_m)
    print(f"[pipeline] Scanning {len(positions)} slices along {axis} axis "
          f"({ax_min:.1f}–{ax_max:.1f} m, step={step_m}m)", flush=True)

    all_gates = []
    t_start = time.time()

    for i, pos in enumerate(positions):
        _, uv, _, _ = extract_slab(pts_z, axis=axis.upper(), position_m=float(pos), thickness_m=thickness_m)
        if uv is None or len(uv) < 10:
            continue
        gates, debug_str = detect_gates(
            uv, axis=axis.upper(), position_m=float(pos), thickness_m=thickness_m,
            pts3d=pts_z,
        )
        if gates:
            print(f"[pipeline]   pos={pos:.1f}m → {len(gates)} gates ({debug_str})", flush=True)
            all_gates.extend([g.to_dict() for g in gates])
        elif (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            pct = (i + 1) / len(positions) * 100
            print(f"[pipeline]   {pct:.0f}% ({i+1}/{len(positions)}) elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"[pipeline] Done — {len(all_gates)} gates found in {elapsed:.0f}s", flush=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        "cloud_path": cloud_path,
        "axis": axis.upper(),
        "step_m": step_m,
        "thickness_m": thickness_m,
        "zmin": zmin,
        "zmax": zmax,
        "n_slices": int(len(positions)),
        "n_gates": len(all_gates),
        "elapsed_s": round(elapsed, 1),
        "gates": all_gates,
    }

    ts = time.strftime("%Y%m%d%H%M")
    gates_json = out_path / f"{ts}_gates.json"
    with open(gates_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[pipeline] Saved → {gates_json}", flush=True)

    # Also save CSV for quick inspection
    import csv
    gates_csv = out_path / f"{ts}_gates.csv"
    if all_gates:
        cols = ["gate_id", "axis", "position_m", "confidence", "pipe_count",
                "opening_area_m2", "bbox_2d"]
        with open(gates_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for g in all_gates:
                row = {k: g[k] for k in cols}
                row["bbox_2d"] = str(g["bbox_2d"])
                w.writerow(row)
        print(f"[pipeline] Saved → {gates_csv}", flush=True)

    if launch:
        _launch_viewer(out_path)

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
    parser.add_argument("--axis", default="auto", choices=["X", "Y", "auto"],
                        help="Scan axis — X, Y, or auto (default: auto)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Slice step in metres (default: 1.0)")
    parser.add_argument("--thickness", type=float, default=0.3,
                        help="Slab thickness in metres (default: 0.3)")
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
