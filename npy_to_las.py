"""npy_to_las.py — Convert a .npy point cloud to LAS format for CloudCompare.

Usage:
    python npy_to_las.py <input.npy> [output.las]

If output path is omitted, saves alongside the input with .las extension.
Expects Nx3 or Nx4+ float array (X Y Z ...).
"""

import sys
import time
from pathlib import Path

import numpy as np
import laspy


def convert(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)

    print(f"Source : {src_path.name}")
    print(f"Output : {dst_path}")

    t_start = time.time()
    print("  Loading ...", end="", flush=True)
    pts = np.load(str(src_path))
    print(f"\r  Loaded {len(pts):,} points  ({time.time()-t_start:.1f}s)")

    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    z = pts[:, 2].astype(np.float64)

    # LAS scale/offset — 1mm precision, offset at cloud min
    scale = 0.001
    offset = [float(x.min()), float(y.min()), float(z.min())]

    header = laspy.LasHeader(point_format=0, version="1.4")
    header.offsets = np.array(offset)
    header.scales  = np.array([scale, scale, scale])

    las = laspy.LasData(header=header)
    las.x = x
    las.y = y
    las.z = z

    print(f"  Writing {dst_path.name} ...", end="", flush=True)
    las.write(str(dst_path))

    size_mb = dst_path.stat().st_size / 1_048_576
    elapsed = time.time() - t_start
    print(f"\r  Done — {len(pts):,} pts  {size_mb:.0f} MB  {elapsed:.1f}s" + " " * 20)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npy_to_las.py <input.npy> [output.las]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else str(Path(src).with_suffix(".las"))
    convert(src, dst)
