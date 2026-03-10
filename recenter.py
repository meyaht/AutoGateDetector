"""recenter.py — translate a point cloud so its min X/Y/Z lands at the origin.

Usage:
    python recenter.py <input.npy> [output.npy]

If output path is omitted, writes <input>_recentered.npy alongside the input.
No rotations — pure translation only.
"""

import sys
from pathlib import Path
import numpy as np


def recenter(input_path: str, output_path: str | None = None) -> Path:
    inp = Path(input_path)
    if output_path is None:
        out = inp.with_name(inp.stem + "_recentered.npy")
    else:
        out = Path(output_path)

    print(f"Loading {inp} ...", flush=True)
    pts = np.load(str(inp))
    print(f"  {len(pts):,} points, shape {pts.shape}", flush=True)

    mins = pts[:, :3].min(axis=0)
    print(f"  min X={mins[0]:.4f}  Y={mins[1]:.4f}  Z={mins[2]:.4f}", flush=True)
    print(f"  Translating by ({-mins[0]:.4f}, {-mins[1]:.4f}, {-mins[2]:.4f}) ...", flush=True)

    pts[:, :3] -= mins

    maxs = pts[:, :3].max(axis=0)
    print(f"  New extents  X=0–{maxs[0]:.3f}m  Y=0–{maxs[1]:.3f}m  Z=0–{maxs[2]:.3f}m", flush=True)

    np.save(str(out), pts)
    print(f"Saved → {out}", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    recenter(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
