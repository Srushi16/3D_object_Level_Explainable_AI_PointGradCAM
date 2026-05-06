#!/usr/bin/env python3
"""
reconstruct_one.py
Reconstruct ONE SonarCloud sample into a 3D point cloud (.npy) using reconstruct_data.py

Run from:
  /netscratch/rsonawane/ex_ai/OpenPCDet/data/sonarcloud/tools
  python3 reconstruct_one.py
"""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image  # pillow

from reconstruct_data import reconstruct3d


def pick_existing(*candidates: str) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(candidates))


def first_file(folder: Path, patterns) -> Path:
    for pat in patterns:
        files = sorted(folder.glob(pat))
        if files:
            return files[0]
    raise FileNotFoundError(f"No files found in {folder} for patterns {patterns}")


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 uint8 RGB."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:  # RGBA -> RGB
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def ensure_2d_depth(depth: np.ndarray) -> np.ndarray:
    """Ensure depth is HxW float32."""
    if depth.ndim == 3:
        # Some TIFFs load as HxWxC; use first channel
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def resize_rgb_to(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize RGB image to (target_h, target_w) using PIL."""
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def main():
    # 1) Pick a sample directory (fix moved_terrain0 naming)
    sample_dir = pick_existing(
        "../boat_data/no_terrain/moved_terrain0/orientation_1",
        "../boat_data/more_ruggedy_terrain/moved_terrain0/orientation_1",
        "../boat_data/straighter_terrain/moved_terrain0/orientation_1",
        "../boat_data/straightest_terrain/moved_terrain0/orientation_1",
    )

    depth_dir = sample_dir / "depth"
    sonar_dir = sample_dir / "sonar_c"

    depth_path = first_file(depth_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    sonar_path = first_file(sonar_dir, ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"])

    print("Using sample_dir:", sample_dir)
    print("Depth file:", depth_path)
    print("Sonar file:", sonar_path)

    # 2) Load
    depth = imageio.imread(depth_path)
    sonar = imageio.imread(sonar_path)

    depth = ensure_2d_depth(depth)
    sonar = ensure_rgb(sonar)

    # 3) IMPORTANT: match sizes (resize sonar -> depth)
    H_d, W_d = depth.shape[:2]
    H_s, W_s = sonar.shape[:2]

    if (H_s != H_d) or (W_s != W_d):
        print(f"[INFO] Resizing sonar from {H_s}x{W_s} -> {H_d}x{W_d} to match depth.")
        sonar = resize_rgb_to(sonar, H_d, W_d)

    # 4) Camera intrinsics (PLACEHOLDER but consistent with resized resolution)
    H, W = sonar.shape[0], sonar.shape[1]
    fx = 300.0
    fy = 300.0
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    # 5) Pose placeholder (ok for qualitative recon sanity)
    x, y, z, yaw = 0.0, 0.0, 0.0, 0.0

    # 6) Reconstruct
    scene, pts = reconstruct3d(
        image=sonar,
        depth_map=depth,
        x=x, y=y, z=z, yaw=yaw,
        camera_parameters=K,
        step=2,
        mesh=False
    )

    pts = np.asarray(pts, dtype=np.float32)
    print("Reconstructed points:", pts.shape)

    # 7) Save
    out_dir = Path("../recon_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"boat_no_terrain_moved_terrain0_orientation_1.npy"
    out_path = out_dir / out_name
    np.save(out_path, pts)
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()
