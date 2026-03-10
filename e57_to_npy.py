"""e57_to_npy.py — Convert an E57 or .npy point cloud to a recentered .npy file.

Translates the cloud so minX/minY/minZ = 0 (all points in positive space).
No rotations — pure translation only.

Usage:
    python e57_to_npy.py <input.e57> [output.npy]
    python e57_to_npy.py <input.npy> [output.npy]

If output path is omitted, saves alongside the input with .npy extension
(e57 input) or <name>_recentered.npy (npy input).
"""

import sys
import time
from pathlib import Path

import numpy as np


def _bar(done: int, total: int, width: int = 32) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _recenter(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = pts[:, :3].min(axis=0)
    pts[:, :3] -= mins
    return pts, mins


def convert(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)

    if src_path.suffix.lower() == ".npy":
        # ── NPY input: just load and recenter ──────────────────────────────
        print(f"Source : {src_path.name}  (npy)")
        print(f"Output : {dst_path}")
        print()
        print(f"  Loading ...", end="", flush=True)
        t_start = time.time()
        pts = np.load(str(src_path))
        total_pts = len(pts)
        print(f"\r  Loaded {total_pts:,} pts  ({time.time()-t_start:.1f}s)")
    else:
        # ── E57 input: read scans ───────────────────────────────────────────
        import pye57
        e57 = pye57.E57(str(src_path))
        n_scans = e57.scan_count

        print(f"Source : {src_path.name}  ({n_scans} scan{'s' if n_scans != 1 else ''})")
        print(f"Output : {dst_path}")
        print()

        chunks = []
        t_start = time.time()
        total_pts = 0

        for idx in range(n_scans):
            elapsed = time.time() - t_start
            pct = int(idx / n_scans * 100)
            pts_str = f"{total_pts:,} pts" if total_pts else ""
            bar = _bar(idx, n_scans)
            print(f"\r  Scan {idx + 1}/{n_scans}  {bar}  {pct:3d}%  {pts_str}  {elapsed:.1f}s  ",
                  end="", flush=True)

            data = e57.read_scan(idx, ignore_missing_fields=True)
            x = np.asarray(data["cartesianX"], dtype=np.float32)
            y = np.asarray(data["cartesianY"], dtype=np.float32)
            z = np.asarray(data["cartesianZ"], dtype=np.float32)
            chunk = np.column_stack([x, y, z])
            chunks.append(chunk)
            total_pts += len(chunk)

        elapsed = time.time() - t_start
        print(f"\r  Scan {n_scans}/{n_scans}  {_bar(n_scans, n_scans)}  100%  {total_pts:,} pts  {elapsed:.1f}s  ")
        print()

        print("  Concatenating ...", end="", flush=True)
        pts = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float32)

    # ── Recenter ────────────────────────────────────────────────────────────
    mins = pts[:, :3].min(axis=0)
    print(f"\r  Recentering — translating by "
          f"({-mins[0]:.3f}, {-mins[1]:.3f}, {-mins[2]:.3f}) m ...", end="", flush=True)
    pts[:, :3] -= mins
    maxs = pts[:, :3].max(axis=0)
    print(f"\r  Extents after recenter: "
          f"X 0–{maxs[0]:.2f}m  Y 0–{maxs[1]:.2f}m  Z 0–{maxs[2]:.2f}m")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"  Saving {total_pts:,} pts → {dst_path.name} ...", end="", flush=True)
    np.save(str(dst_path), pts)

    size_mb = dst_path.stat().st_size / 1_048_576
    elapsed = time.time() - t_start
    print(f"\r  Done — {total_pts:,} pts  {size_mb:.0f} MB  {elapsed:.1f}s" + " " * 30)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python e57_to_npy.py <input.e57|npy> [output.npy]")
        sys.exit(1)

    src = sys.argv[1]
    src_path = Path(src)
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    elif src_path.suffix.lower() == ".npy":
        dst = str(src_path.with_name(src_path.stem + "_recentered.npy"))
    else:
        dst = str(src_path.with_suffix(".npy"))

    convert(src, dst)
