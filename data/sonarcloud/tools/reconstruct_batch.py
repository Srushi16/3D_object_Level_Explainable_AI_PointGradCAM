#!/usr/bin/env python3
"""
reconstruct_batch.py (Option 1: raw folders)

Run from:
  /netscratch/rsonawane/ex_ai/OpenPCDet/data/sonarcloud/tools

It will:
- reconstruct a small curated set of SonarCloud samples into point clouds (.npy)
- save outputs in ../recon_out/
- write ../recon_out/recon_summary.csv

Edit the SAMPLES list below to add/remove cases.
"""

from pathlib import Path
import csv
import numpy as np
import imageio.v2 as imageio
from PIL import Image

from reconstruct_data import reconstruct3d


# ---------------------------------------------------------------------
# CONFIG: curated "best samples" (few only)
# Paths are relative to tools/ directory
# ---------------------------------------------------------------------
SAMPLES = [
    # ---------------- Boat ----------------
    {"name": "boat_noTerrain_ori1",
     "sample_dir": "../boat_data/no_terrain/moved_terrain0/orientation_1"},

    # Optional: one harder terrain case
    {"name": "boat_rugged_ori1",
     "sample_dir": "../boat_data/more_ruggedy_terrain/moved_terrain0/orientation_1"},

    # ---------------- UXO Big ----------------
    # Use 1–2 rotations only (effective, low effort)
    {"name": "uxoBig_noTerrain_rot0_0",
     "sample_dir": "../uxo_big_data/no_terrain/moved_terrain0/rot0_0"},

    {"name": "uxoBig_noTerrain_rot90_0",
     "sample_dir": "../uxo_big_data/no_terrain/moved_terrain0/rot90_0"},

    # ---------------- Shapes ----------------
    {"name": "cylinder_noTerrain_ori1",
     "sample_dir": "../cylinder_data/no_terrain/moved_terrain0/orientation_1"},

    {"name": "sphere_noTerrain_ori1",
     "sample_dir": "../sphere_data/no_terrain/moved_terrain0/orientation_1"},

    {"name": "cube_noTerrain_ori1",
     "sample_dir": "../cube_data/no_terrain/moved_terrain0/orientation_1"},
]

# Reconstruction controls
STEP = 2         # try 8; if too dense, use 10 or 12
MESH = False

# Intrinsics placeholder (consistent, not physically exact)
FX = 300.0
FY = 300.0

# Pose placeholder (neutral); OK for your qualitative XAI dataset creation
POSE_X, POSE_Y, POSE_Z, POSE_YAW = 0.0, 0.0, 0.0, 0.0


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def ensure_2d_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def resize_rgb_to(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def first_file(folder: Path, patterns) -> Path:
    for pat in patterns:
        files = sorted(folder.glob(pat))
        if files:
            return files[0]
    raise FileNotFoundError(f"No files found in {folder} for patterns {patterns}")


def minimal_cleanup(pts: np.ndarray) -> np.ndarray:
    """Keep only finite points and remove near-zero points."""
    mask = np.isfinite(pts).all(axis=1)
    mask &= (np.linalg.norm(pts, axis=1) > 1e-6)
    return pts[mask].astype(np.float32)


def reconstruct_one(sample_name: str, sample_dir: Path, out_dir: Path):
    depth_dir = sample_dir / "depth"
    sonar_dir = sample_dir / "sonar_c"

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_dir}")
    if not sonar_dir.exists():
        raise FileNotFoundError(f"Sonar dir not found: {sonar_dir}")

    depth_path = first_file(depth_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    sonar_path = first_file(sonar_dir, ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"])

    depth = ensure_2d_depth(imageio.imread(depth_path))
    sonar = ensure_rgb(imageio.imread(sonar_path))

    # Match resolutions (required)
    H_d, W_d = depth.shape[:2]
    H_s, W_s = sonar.shape[:2]
    if (H_s != H_d) or (W_s != W_d):
        sonar = resize_rgb_to(sonar, H_d, W_d)

    # Intrinsics consistent with final image size
    H, W = sonar.shape[0], sonar.shape[1]
    K = np.array([[FX, 0, W / 2.0],
                  [0, FY, H / 2.0],
                  [0,  0,    1.0]], dtype=np.float32)

    # Reconstruct
    _, pts_list = reconstruct3d(
        image=sonar,
        depth_map=depth,
        x=POSE_X, y=POSE_Y, z=POSE_Z, yaw=POSE_YAW,
        camera_parameters=K,
        step=STEP,
        mesh=MESH
    )

    pts = np.asarray(pts_list, dtype=np.float32)  # (N,3)
    pts = minimal_cleanup(pts)

    out_path = out_dir / f"{sample_name}.npy"
    np.save(out_path, pts)

    return {
        "name": sample_name,
        "sample_dir": str(sample_dir),
        "depth_file": str(depth_path),
        "sonar_file": str(sonar_path),
        "num_points": int(pts.shape[0]),
        "min_x": float(np.min(pts[:, 0])),
        "max_x": float(np.max(pts[:, 0])),
        "min_y": float(np.min(pts[:, 1])),
        "max_y": float(np.max(pts[:, 1])),
        "min_z": float(np.min(pts[:, 2])),
        "max_z": float(np.max(pts[:, 2])),
        "saved_npy": str(out_path),
    }


def main():
    out_dir = Path("../recon_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "recon_summary.csv"
    rows = []

    print(f"[INFO] Writing outputs to: {out_dir.resolve()}")
    print(f"[INFO] STEP={STEP}, MESH={MESH}")

    for s in SAMPLES:
        name = s["name"]
        sample_dir = Path(s["sample_dir"])

        print(f"\n[RUN] {name}")
        print(f"  dir: {sample_dir}")

        try:
            row = reconstruct_one(name, sample_dir, out_dir)
            rows.append(row)
            print(f"  -> saved {row['saved_npy']}  (N={row['num_points']})")
        except Exception as e:
            print(f"  !! FAILED: {e}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[INFO] Summary CSV saved: {summary_csv.resolve()}")
    else:
        print("\n[WARN] No reconstructions succeeded; summary not written.")


if __name__ == "__main__":
    main()
