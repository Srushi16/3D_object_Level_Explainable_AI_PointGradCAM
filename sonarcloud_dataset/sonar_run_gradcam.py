#!/usr/bin/env python3
"""
PointGrad-CAM runner for SonarCloud using OpenPCDet (PointRCNN).

Purpose
  - Compute point-wise Grad-CAM saliency for SONAR point clouds by running a KITTI-trained PointRCNN.
  - SONAR points (N,3) are transformed into KITTI-like LiDAR points (N,4) via scale/shift + synthetic intensity.
  - Exports colored point clouds (PLY/NPZ) and debugging metadata for reproducible analysis.

Modes
  (A) Objectness Grad-CAM (default, recommended for SONAR):
      - Explain the strongest ROI logit (max over ROIs), optionally restricted to a snapped target region.
      - Produces ONE explanation per sample.

  (B) Per-detection Grad-CAM (--per_detect):
      - Run post-processing once to get final detections.
      - For each detection, match it to a ROI and explain that ROI logit.
      - Produces multiple explanations per sample.

Outputs
  - PLY + NPZ with point-wise heatmap
  - debug.npz with captured tensor shapes and ROI selection details
  - metadata.json (+ detections.csv in per-detection mode)

Notes
  - Uses pre-NMS logits from roi_head.cls_layers.7 (differentiable).
  - Maps pooled CAM -> input points via KDTree (radius Gaussian or kNN).
"""

import os
import sys
import argparse
import random
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# UPDATE PATH (if needed): ensure this points to the OpenPCDet repo root so `import pcdet` works

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

try:
    from pcdet.datasets.kitti.kitti_zip_dataset import KittiZipDataset
except Exception:
    KittiZipDataset = None

from pointgradCAM.gradcam import PointGradCAM
from pointgradCAM.visualize import colorize_points, save_colored_ply, save_colored_npz

IOU_UTIL = None
try:
    from pcdet.ops.iou3d_nms import iou3d_nms_utils as IOU_UTIL  # type: ignore
except Exception:
    IOU_UTIL = None


def load_sonar_points(
    npy_path: str,
    cap_points: int = 10000,
    intensity_mode: str = "uniform_high",
    intensity_value: float = 0.7,
    seed: int = 42,
    xyz_scale: float = 5.0,
    xyz_shift: Tuple[float, float, float] = (25.0, 0.0, -1.5),
) -> np.ndarray:
    """
    Loads sonar (N,3) from npy and returns (N,4) [x,y,z,intensity] after scale/shift.
    Downsamples to cap_points if needed (no upsample here; fixed-N handled later).
    """
    p = Path(npy_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing sonar npy: {npy_path}")

    pts = np.load(p).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected sonar npy shape (N,3+) but got {pts.shape}")

    xyz = pts[:, :3].astype(np.float32)
    xyz = xyz * float(xyz_scale)
    xyz = xyz + np.array(xyz_shift, dtype=np.float32)

    rng = np.random.default_rng(seed)
    if intensity_mode == "random":
        inten = rng.uniform(0.1, 0.9, (xyz.shape[0], 1)).astype(np.float32)
    elif intensity_mode == "const":
        inten = np.full((xyz.shape[0], 1), float(intensity_value), dtype=np.float32)
    else:
        # uniform_high
        inten = np.full((xyz.shape[0], 1), 0.7, dtype=np.float32)

    pts4 = np.hstack([xyz, inten]).astype(np.float32)

    finite = np.isfinite(pts4).all(axis=1)
    pts4 = pts4[finite]
    nonzero = np.linalg.norm(pts4[:, :3], axis=1) > 1e-6
    pts4 = pts4[nonzero]

    if cap_points is not None and pts4.shape[0] > int(cap_points):
        idx = rng.choice(pts4.shape[0], size=int(cap_points), replace=False)
        pts4 = pts4[idx]

    return pts4


def pad_or_downsample_fixed_n(pts4: np.ndarray, fixed_n: int, seed: int = 42) -> np.ndarray:
    """
    Make exactly fixed_n points:
      - if > fixed_n: downsample without replacement
      - if < fixed_n: duplicate with replacement
    """
    rng = np.random.default_rng(seed)
    N = pts4.shape[0]
    if N == fixed_n:
        return pts4
    if N > fixed_n:
        idx = rng.choice(N, fixed_n, replace=False)
        return pts4[idx]
    # N < fixed_n
    extra = rng.choice(N, fixed_n - N, replace=True)
    return np.concatenate([pts4, pts4[extra]], axis=0)


def load_snapped_region_from_csv(
    recon_summary_csv: Optional[str],
    sample_tag: str,
    xyz_scale: float,
    xyz_shift: Tuple[float, float, float]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (snapped_center, snapped_lwh) in transformed coords.
    """
    if recon_summary_csv is None:
        return None, None
    p = Path(recon_summary_csv)
    if not p.exists():
        return None, None

    import pandas as pd
    df = pd.read_csv(p)
    row = df[df["name"] == sample_tag]
    if len(row) == 0:
        return None, None

    obj_min = row[["min_x", "min_y", "min_z"]].values[0].astype(np.float32)
    obj_max = row[["max_x", "max_y", "max_z"]].values[0].astype(np.float32)

    obj_min = obj_min * float(xyz_scale) + np.array(xyz_shift, dtype=np.float32)
    obj_max = obj_max * float(xyz_scale) + np.array(xyz_shift, dtype=np.float32)

    center = (obj_min + obj_max) / 2.0
    lwh = (obj_max - obj_min)
    return center.astype(np.float32), lwh.astype(np.float32)


def filter_rois_in_region(rois_np: np.ndarray,
                          target_region: str,
                          target_center: Optional[np.ndarray],
                          target_lwh: Optional[np.ndarray],
                          region_expand: float,
                          region_radius: float) -> np.ndarray:
    """
    Returns mask over ROIs (R,) indicating which ROIs are inside region.
    rois_np: [R,7]
    """
    R = rois_np.shape[0]
    keep = np.ones((R,), dtype=bool)
    if target_region == "global" or target_center is None:
        return keep

    c = rois_np[:, :3]
    if target_region == "radius":
        d = np.linalg.norm(c - target_center[None, :], axis=1)
        keep &= (d <= float(region_radius))
    else:
        # box
        half = 0.5 * target_lwh * float(region_expand)
        lo = target_center - half
        hi = target_center + half
        keep &= np.all((c >= lo[None, :]) & (c <= hi[None, :]), axis=1)

    return keep


def list_relevant_modules(model: torch.nn.Module) -> None:
    print("\n=== Relevant modules (roi/sa/fp/backbone/point_head) ===")
    for name, _ in model.named_modules():
        low = name.lower()
        if any(k in low for k in ["roi", "sa_modules", "fp_modules", "backbone_3d", "point_head"]):
            print(name)


def best_roi_match(rois_lidar: np.ndarray, det_box_lidar: np.ndarray) -> int:
    """Return ROI index matching a selected final detection box (x,y,z,dx,dy,dz,yaw)."""
    if rois_lidar.ndim != 2 or rois_lidar.shape[1] != 7:
        centers = rois_lidar.reshape(-1, 7)[:, :3]
        d = np.linalg.norm(centers - det_box_lidar[:3][None, :], axis=1)
        return int(np.argmin(d))

    if IOU_UTIL is not None:
        try:
            rois_t = torch.from_numpy(rois_lidar).float().cpu()
            det_t = torch.from_numpy(det_box_lidar[None, :]).float().cpu()
            if hasattr(IOU_UTIL, "boxes_bev_iou_cpu"):
                ious = IOU_UTIL.boxes_bev_iou_cpu(rois_t, det_t).numpy().reshape(-1)
            else:
                ious = IOU_UTIL.boxes_iou_bev(rois_t, det_t).numpy().reshape(-1)
            return int(np.argmax(ious))
        except Exception:
            pass

    centers = rois_lidar[:, :3]
    d = np.linalg.norm(centers - det_box_lidar[:3][None, :], axis=1)
    return int(np.argmin(d))


def canonical_to_world(pooled_xyz_roi: np.ndarray, roi_box: np.ndarray) -> np.ndarray:
    cx, cy, cz, _, _, _, yaw = roi_box.astype(np.float32)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    return (pooled_xyz_roi @ R.T) + np.array([cx, cy, cz], dtype=np.float32)[None, :]


def detect_pooled_frame(xyz_roi: np.ndarray, roi_box: np.ndarray) -> str:
    center = roi_box[:3].astype(np.float32)
    mean_xyz = xyz_roi.mean(axis=0).astype(np.float32)

    dist_to_center = float(np.linalg.norm(mean_xyz - center))
    dist_to_zero = float(np.linalg.norm(mean_xyz))

    dx, dy, dz = roi_box[3:6].astype(np.float32)
    canonical_like = (dist_to_zero < 5.0) and (np.abs(mean_xyz[0]) < max(5.0, dx * 2)) and (np.abs(mean_xyz[1]) < max(5.0, dy * 2))
    world_like = dist_to_center < 3.0

    if world_like and not canonical_like:
        return "world"
    if canonical_like and not world_like:
        return "canonical"
    return "world"


def scatter_pooled_xyz_to_input_knn(input_xyz, pooled_xyz, pooled_scores, k=20):
    from scipy.spatial import cKDTree
    tree = cKDTree(input_xyz)
    _, idx = tree.query(pooled_xyz, k=max(1, k))

    out = np.zeros(input_xyz.shape[0], dtype=np.float32)

    if k <= 1:
        for j, s in zip(idx.astype(np.int64), pooled_scores):
            out[j] += float(s)
    else:
        idx = np.asarray(idx).astype(np.int64)
        for row, s in zip(idx, pooled_scores):
            w = float(s) / float(len(row))
            for j in row:
                out[j] += w
    return out


def scatter_pooled_xyz_to_input_radius(
    input_xyz: np.ndarray,
    pooled_xyz: np.ndarray,
    pooled_scores: np.ndarray,
    radius: float = 0.50,
    sigma: float = 0.22,
    max_neighbors: int = 256,
    agg: str = "sum",
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(input_xyz)
    out = np.zeros(input_xyz.shape[0], dtype=np.float32)
    s2 = max(1e-6, sigma * sigma)

    for p, s in zip(pooled_xyz, pooled_scores):
        s = float(s)
        if s <= 0:
            continue

        idx = tree.query_ball_point(p, radius)
        if not idx:
            continue

        pts = input_xyz[idx]
        d2 = np.sum((pts - p[None, :]) ** 2, axis=1)

        if len(idx) > max_neighbors:
            sel = np.argsort(d2)[:max_neighbors]
            idx = [idx[i] for i in sel]
            d2 = d2[sel]

        w = np.exp(-d2 / (2.0 * s2)).astype(np.float32)
        contrib = (s * w).astype(np.float32)

        if agg == "max":
            for j, v in zip(idx, contrib):
                if v > out[j]:
                    out[j] = v
        else:
            out[idx] += contrib

    return out


def safe_score_tag(score: float) -> str:
    return f"{score:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser("PointGrad-CAM for SonarCloud (PointRCNN / OpenPCDet)")
    parser.add_argument("--cfg", required=True)
    # UPDATE PATH: --cfg should point to your PointRCNN config YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    parser.add_argument("--ckpt", required=True)
    # UPDATE PATH: --ckpt should point to your trained checkpoint (.pth)

    # Still need KITTI dataset metadata to build model
    parser.add_argument("--root_path", required=True)
    # UPDATE PATH: --root_path should point to the KITTI dataset root (needed for OpenPCDet metadata)
    parser.add_argument("--info_dir", required=True)
    # UPDATE PATH: --info_dir should point to the folder containing KITTI info files (kitti_infos_*.pkl)

    # Sonar
    parser.add_argument("--sonar_npy", required=True)
    # UPDATE PATH: --sonar_npy should point to your SONAR point cloud .npy file (shape (N,3) or (N,3+))
    parser.add_argument("--outdir", default="./gradcam_sonar_outputs")
    # UPDATE PATH: --outdir is the output folder where Grad-CAM results will be written
    parser.add_argument("--list-modules", action="store_true")

    # Transform knobs
    parser.add_argument("--xyz_scale", type=float, default=5.0)
    parser.add_argument("--xyz_shift", type=float, nargs=3, default=[25.0, 0.0, -1.5])
    parser.add_argument("--intensity_mode", choices=["uniform_high", "const", "random"], default="uniform_high")
    parser.add_argument("--sonar_intensity", type=float, default=0.7)
    parser.add_argument("--cap_points", type=int, default=10000)

    # Fixed N
    parser.add_argument("--fixed_n", type=int, default=10000)

    # Region (snapped)
    parser.add_argument("--recon_summary_csv", type=str, default=None)
    # UPDATE PATH (optional): CSV with snapped bounds (min/max) per sample to restrict ROI selection
    parser.add_argument("--target_region", choices=["global", "box", "radius"], default="global")
    parser.add_argument("--region_expand", type=float, default=1.3)
    parser.add_argument("--region_radius", type=float, default=6.0)

    # CAM hook
    parser.add_argument("--act-layer", default="roi_head.merge_down_layer")
    parser.add_argument("--verbose", action="store_true")

    # Mode
    parser.add_argument("--per_detect", action="store_true",
                        help="If set, export one explanation per final detection. Otherwise export ONE objectness explanation.")
    parser.add_argument("--min_score", type=float, default=1e-6)
    parser.add_argument("--max_dets", type=int, default=50)

    # Mapping
    parser.add_argument("--map-mode", default="radius", choices=["radius", "knn"])
    parser.add_argument("--map-k", type=int, default=20)
    parser.add_argument("--map-radius", type=float, default=0.50)
    parser.add_argument("--map-sigma", type=float, default=0.22)
    parser.add_argument("--map-maxn", type=int, default=256)
    parser.add_argument("--map-agg", default="sum", choices=["sum", "max"])

    parser.add_argument("--clip-percentile", type=float, default=99.5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config & build model with KITTI dataset metadata
    cfg_from_yaml_file(args.cfg, cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = float(args.min_score)
    cfg.DATA_CONFIG.DATA_PATH = args.root_path
    cfg.DATA_CONFIG.INFO_DIR = args.info_dir

    logger = common_utils.create_logger()

    from pcdet.datasets.kitti.kitti_zip_dataset import KittiZipDataset
    dataset = KittiZipDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.root_path),
        logger=logger
    )

    print(f"[OK] Building model from {args.cfg} on {device} ...")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.to(device)
    model.eval()

    if args.list_modules:
        list_relevant_modules(model)
        return

    torch.set_grad_enabled(True)

    # Load sonar points
    sample_tag = Path(args.sonar_npy).stem
    pts4 = load_sonar_points(
        args.sonar_npy,
        cap_points=int(args.cap_points),
        intensity_mode=args.intensity_mode,
        intensity_value=float(args.sonar_intensity),
        seed=SEED,
        xyz_scale=float(args.xyz_scale),
        xyz_shift=(float(args.xyz_shift[0]), float(args.xyz_shift[1]), float(args.xyz_shift[2])),
    )
    pts4 = pad_or_downsample_fixed_n(pts4, int(args.fixed_n), seed=SEED)

    input_xyz = pts4[:, :3]
    pts_with_batch = np.hstack([np.zeros((pts4.shape[0], 1), dtype=np.float32), pts4]).astype(np.float32)
    points_tensor = torch.from_numpy(pts_with_batch).to(device)

    def make_input_dict():
        return {"batch_size": 1, "points": points_tensor}

    print(f"[DATA] sonar points: {pts4.shape} (after transform + fixed_n)")
    print(f"[DATA] xyz min={input_xyz.min(axis=0).round(3)} max={input_xyz.max(axis=0).round(3)} mean={input_xyz.mean(axis=0).round(3)}")

    # Snapped region (optional)
    snapped_center, snapped_lwh = load_snapped_region_from_csv(
        args.recon_summary_csv,
        sample_tag=sample_tag,
        xyz_scale=float(args.xyz_scale),
        xyz_shift=(float(args.xyz_shift[0]), float(args.xyz_shift[1]), float(args.xyz_shift[2])),
    )

    if snapped_center is not None:
        print(f"[SNAP] center={snapped_center.round(3)} lwh={snapped_lwh.round(3)}")
        print(f"[TARGET] target_region={args.target_region} expand={args.region_expand} radius={args.region_radius}")
    else:
        print("[SNAP] no snapped bounds found -> using global")

    # Capture: ROIs, pooled xyz, logits
    capture: Dict[str, Any] = {"rois": None, "pooled_xyz": None, "rcnn_logits": None}

    roipool_name = "roi_head.roipoint_pool3d_layer"
    roipool_mod = dict(model.named_modules()).get(roipool_name, None)
    if roipool_mod is None:
        print(f"[WARN] {roipool_name} not found; mapping may degrade.")
    else:
        def roipool_hook(m, inp, out):
            # inp[2]: [B,R,7]
            try:
                rois = inp[2]
                if torch.is_tensor(rois) and rois.dim() == 3 and rois.shape[-1] == 7:
                    capture["rois"] = rois
            except Exception:
                pass

            # out[0]: [B,R,P,C’], xyz is first 3
            try:
                if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                    pooled = out[0]
                    if pooled.dim() == 4 and pooled.shape[-1] >= 3:
                        capture["pooled_xyz"] = pooled[..., 0:3].squeeze(0).contiguous()  # [R,P,3]
            except Exception:
                pass

        roipool_mod.register_forward_hook(roipool_hook)
        print(f"[OK] Hooked {roipool_name} to capture rois/pooled_xyz")

    cls7_name = "roi_head.cls_layers.7"
    cls7_mod = dict(model.named_modules()).get(cls7_name, None)
    if cls7_mod is None:
        raise RuntimeError(f"{cls7_name} not found.")
    else:
        def cls7_hook(m, inp, out):
            capture["rcnn_logits"] = out
        cls7_mod.register_forward_hook(cls7_hook)
        print(f"[OK] Hooked {cls7_name} to capture pre-NMS rcnn logits")

    # Run once to list detections (post-processed)
    with torch.no_grad():
        outputs_once = model(make_input_dict())

    pred_dicts_once = outputs_once[0] if isinstance(outputs_once, (list, tuple)) else outputs_once
    pred0 = pred_dicts_once[0]

    boxes_all = pred0["pred_boxes"].detach().cpu().numpy() if "pred_boxes" in pred0 else np.zeros((0, 7), dtype=np.float32)
    scores_all = pred0["pred_scores"].detach().cpu().numpy() if "pred_scores" in pred0 else np.zeros((0,), dtype=np.float32)
    labels_all = pred0["pred_labels"].detach().cpu().numpy() if "pred_labels" in pred0 else np.zeros((0,), dtype=np.int32)

    print(f"[PRED] postproc detections: {len(scores_all)}")
    if len(scores_all) > 0:
        order = np.argsort(-scores_all)[:min(10, len(scores_all))]
        for r, i in enumerate(order, 1):
            lab = int(labels_all[i])
            cls = cfg.CLASS_NAMES[lab - 1] if 1 <= lab <= len(cfg.CLASS_NAMES) else f"cls{lab}"
            b = boxes_all[i]
            print(f"[TOP] {r:02d}: score={scores_all[i]:.6f} cls={cls} center=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f}) lwh=({b[3]:.2f},{b[4]:.2f},{b[5]:.2f}) yaw={b[6]:.2f}")

    # Prepare output folders
    sample_dir = Path(args.outdir) / sample_tag
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Grad-CAM object
    cam = PointGradCAM(model, args.act_layer, device=device)
    cam.verbose = args.verbose

    # Target function helpers
    def _get_rcnn_logits_2d() -> torch.Tensor:
        if capture["rcnn_logits"] is None:
            raise RuntimeError("rcnn_logits not captured (cls_layers.7 hook did not fire).")
        rcnn_logits = capture["rcnn_logits"]
        # normalize common shapes -> [R, C]
        if rcnn_logits.dim() == 3 and rcnn_logits.shape[-1] == 1:
            rcnn_logits = rcnn_logits.squeeze(-1)
        if rcnn_logits.dim() == 3 and rcnn_logits.shape[1] == 1:
            rcnn_logits = rcnn_logits.squeeze(1)
        if rcnn_logits.dim() != 2:
            raise RuntimeError(f"Unexpected rcnn_logits shape after squeeze: {tuple(rcnn_logits.shape)}")
        return rcnn_logits

    # MODE A: Objectness Grad-CAM (ONE output)
    if not args.per_detect:
        holder = {"roi_idx": -1, "target_logit": 0.0, "pooled_frame": "unknown", "roi_keep_count": 0}

        def target_fn(outputs):
            # outputs needed only so that forward executed; ROI logits from hook
            _ = outputs
            rcnn_logits = _get_rcnn_logits_2d()  # [R,C]
            R = rcnn_logits.shape[0]

            if capture["rois"] is None:
                # fallback: global max logit
                roi_idx = int(torch.argmax(rcnn_logits.max(dim=1).values).item())
                holder["roi_idx"] = roi_idx
                t = rcnn_logits[roi_idx].max()
                holder["target_logit"] = float(t.detach().cpu().item())
                return t

            rois_np = capture["rois"].detach().cpu().numpy().squeeze(0)  # [R,7]
            keep = filter_rois_in_region(
                rois_np,
                target_region=args.target_region,
                target_center=snapped_center,
                target_lwh=snapped_lwh if snapped_lwh is not None else None,
                region_expand=float(args.region_expand),
                region_radius=float(args.region_radius),
            )
            holder["roi_keep_count"] = int(keep.sum())
            if keep.sum() == 0:
                # fallback to global max if region excluded everything
                roi_idx = int(torch.argmax(rcnn_logits.max(dim=1).values).item())
            else:
                # best ROI among kept
                vals = rcnn_logits.max(dim=1).values  # [R]
                vals_kept = vals.clone()
                vals_kept[torch.from_numpy(~keep).to(vals_kept.device)] = -1e9
                roi_idx = int(torch.argmax(vals_kept).item())

            holder["roi_idx"] = int(roi_idx)
            t = rcnn_logits[roi_idx].max()
            holder["target_logit"] = float(t.detach().cpu().item())
            return t

        cam_list = cam.attribute(make_input_dict(), target_fn=target_fn, retain_graph=False)
        roi_idx_used = int(np.clip(holder["roi_idx"], 0, len(cam_list) - 1))
        pooled_cam = np.asarray(cam_list[roi_idx_used], dtype=np.float32)

        print(f"[OBJ] selected_roi={roi_idx_used} target_logit={holder['target_logit']:.6f} keep_rois_in_region={holder['roi_keep_count']}")

        # Map pooled CAM -> input points
        point_scores = np.zeros(input_xyz.shape[0], dtype=np.float32)
        pooled_frame = "missing"

        if capture["pooled_xyz"] is None or capture["rois"] is None:
            print("[WARN] pooled_xyz/rois missing -> cannot map.")
        else:
            pooled_xyz_np = capture["pooled_xyz"].detach().cpu().numpy()     # [R,P,3]
            rois_np = capture["rois"].detach().cpu().numpy().squeeze(0)      # [R,7]

            xyz_roi = pooled_xyz_np[roi_idx_used]                            # [P,3]
            roi_box = rois_np[roi_idx_used]                                  # [7]

            frame = detect_pooled_frame(xyz_roi, roi_box)
            pooled_frame = frame

            if frame == "canonical":
                xyz_roi_world = canonical_to_world(xyz_roi, roi_box)
            else:
                xyz_roi_world = xyz_roi

            P = min(xyz_roi_world.shape[0], pooled_cam.shape[0])
            xyz_roi_world = xyz_roi_world[:P]
            pooled_cam_use = pooled_cam[:P]

            if args.map_mode == "knn":
                point_scores = scatter_pooled_xyz_to_input_knn(input_xyz, xyz_roi_world, pooled_cam_use, k=max(1, args.map_k))
            else:
                point_scores = scatter_pooled_xyz_to_input_radius(
                    input_xyz, xyz_roi_world, pooled_cam_use,
                    radius=float(args.map_radius),
                    sigma=float(args.map_sigma),
                    max_neighbors=int(args.map_maxn),
                    agg=str(args.map_agg),
                )

        nz = point_scores[point_scores > 0]
        if nz.size > 0:
            vmax = float(np.percentile(nz, float(args.clip_percentile)))
        else:
            vmax = float(point_scores.max()) if float(point_scores.max()) > 0 else 1.0
        if vmax <= 1e-9:
            vmax = 1.0

        scores = np.clip(point_scores, 0, vmax) / (vmax + 1e-9)

        colors = colorize_points(input_xyz, scores)

        ply_path = sample_dir / "gradcam_objectness.ply"
        npz_path = sample_dir / "gradcam_objectness.npz"
        dbg_path = sample_dir / "gradcam_objectness_debug.npz"

        save_colored_ply(str(ply_path), input_xyz, colors)
        save_colored_npz(str(npz_path), input_xyz, colors, scores=scores)

        np.savez_compressed(
            dbg_path,
            sample=np.array([sample_tag], dtype=object),
            act_layer=np.array([args.act_layer], dtype=object),
            selected_roi=np.array([roi_idx_used], dtype=np.int32),
            target_logit=np.array([holder["target_logit"]], dtype=np.float32),
            pooled_frame=np.array([pooled_frame], dtype=object),
            rois_shape=np.array(list(capture["rois"].shape), dtype=np.int32) if capture["rois"] is not None else np.array([-1], dtype=np.int32),
            pooled_xyz_shape=np.array(list(capture["pooled_xyz"].shape), dtype=np.int32) if capture["pooled_xyz"] is not None else np.array([-1], dtype=np.int32),
            rcnn_logits_shape=np.array(list(capture["rcnn_logits"].shape), dtype=np.int32) if capture["rcnn_logits"] is not None else np.array([-1], dtype=np.int32),
            keep_rois_in_region=np.array([holder["roi_keep_count"]], dtype=np.int32),
            target_region=np.array([args.target_region], dtype=object),
            region_expand=np.array([args.region_expand], dtype=np.float32),
            region_radius=np.array([args.region_radius], dtype=np.float32),
            snapped_center=snapped_center if snapped_center is not None else np.zeros((3,), dtype=np.float32),
            snapped_lwh=snapped_lwh if snapped_lwh is not None else np.zeros((3,), dtype=np.float32),
            map_mode=np.array([args.map_mode], dtype=object),
            map_k=np.array([args.map_k], dtype=np.int32),
            map_radius=np.array([args.map_radius], dtype=np.float32),
            map_sigma=np.array([args.map_sigma], dtype=np.float32),
            map_maxn=np.array([args.map_maxn], dtype=np.int32),
            map_agg=np.array([args.map_agg], dtype=object),
            clip_percentile=np.array([args.clip_percentile], dtype=np.float32),
        )

        meta = {
            "sample": sample_tag,
            "sonar_npy": str(Path(args.sonar_npy).resolve()),
            "mode": "objectness",
            "act_layer": args.act_layer,
            "selected_roi": int(roi_idx_used),
            "target_logit": float(holder["target_logit"]),
            "keep_rois_in_region": int(holder["roi_keep_count"]),
            "target_region": args.target_region,
            "region_expand": float(args.region_expand),
            "region_radius": float(args.region_radius),
            "mapping": {
                "mode": args.map_mode,
                "k": int(args.map_k),
                "radius": float(args.map_radius),
                "sigma": float(args.map_sigma),
                "max_neighbors": int(args.map_maxn),
                "agg": str(args.map_agg),
                "clip_percentile": float(args.clip_percentile),
            },
            "outputs": {
                "ply": "gradcam_objectness.ply",
                "npz": "gradcam_objectness.npz",
                "debug": "gradcam_objectness_debug.npz",
            }
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Saved objectness Grad-CAM -> {sample_dir}")
        cam.remove_hooks()
        return

    # MODE B: Per-detection (optional)
    # filter detections
    order = np.argsort(-scores_all)
    order = order[:min(len(order), int(args.max_dets))]

    det_candidates = [int(i) for i in order if float(scores_all[i]) >= float(args.min_score)]
    print(f"[PER_DET] explaining {len(det_candidates)} detections (min_score={args.min_score})")

    per_class_dir: Dict[str, Path] = {cls: (sample_dir / cls) for cls in cfg.CLASS_NAMES}
    for p in per_class_dir.values():
        p.mkdir(parents=True, exist_ok=True)

    detection_meta_list: List[Dict[str, Any]] = []

    csv_path = sample_dir / "detections.csv"
    csv_fields = [
        "sample", "det_idx", "class_name", "score",
        "roi_idx", "target_logit", "pooled_frame",
        "folder", "npz", "ply", "dbg"
    ]

    for det_idx in det_candidates:
        det_box = boxes_all[det_idx][:7].astype(np.float32)
        det_score = float(scores_all[det_idx])
        det_label = int(labels_all[det_idx])
        class_name = cfg.CLASS_NAMES[det_label - 1] if 1 <= det_label <= len(cfg.CLASS_NAMES) else f"cls{det_label}"

        det_dir = per_class_dir.get(class_name, sample_dir / class_name) / f"det{det_idx:03d}_score{safe_score_tag(det_score)}"
        det_dir.mkdir(parents=True, exist_ok=True)

        holder = {"roi_idx": 0, "target_logit": 0.0, "pooled_frame": "unknown"}

        def target_fn(outputs):
            _ = outputs
            rcnn_logits = _get_rcnn_logits_2d()  # [R,C]
            if capture["rois"] is not None:
                rois_np = capture["rois"].detach().cpu().numpy().squeeze(0)
                roi_idx = best_roi_match(rois_np, det_box)
            else:
                roi_idx = int(torch.argmax(rcnn_logits.max(dim=1).values).item())

            holder["roi_idx"] = int(roi_idx)
            t = rcnn_logits[roi_idx].max()
            holder["target_logit"] = float(t.detach().cpu().item())
            return t

        cam_list = cam.attribute(make_input_dict(), target_fn=target_fn, retain_graph=False)
        roi_idx_used = int(np.clip(holder["roi_idx"], 0, len(cam_list) - 1))
        pooled_cam = np.asarray(cam_list[roi_idx_used], dtype=np.float32)

        point_scores = np.zeros(input_xyz.shape[0], dtype=np.float32)
        pooled_frame = "missing"

        if capture["pooled_xyz"] is not None and capture["rois"] is not None:
            pooled_xyz_np = capture["pooled_xyz"].detach().cpu().numpy()
            rois_np = capture["rois"].detach().cpu().numpy().squeeze(0)

            xyz_roi = pooled_xyz_np[roi_idx_used]
            roi_box = rois_np[roi_idx_used]

            frame = detect_pooled_frame(xyz_roi, roi_box)
            pooled_frame = frame
            holder["pooled_frame"] = frame

            xyz_roi_world = canonical_to_world(xyz_roi, roi_box) if frame == "canonical" else xyz_roi

            P = min(xyz_roi_world.shape[0], pooled_cam.shape[0])
            xyz_roi_world = xyz_roi_world[:P]
            pooled_cam_use = pooled_cam[:P]

            if args.map_mode == "knn":
                point_scores = scatter_pooled_xyz_to_input_knn(input_xyz, xyz_roi_world, pooled_cam_use, k=max(1, args.map_k))
            else:
                point_scores = scatter_pooled_xyz_to_input_radius(
                    input_xyz, xyz_roi_world, pooled_cam_use,
                    radius=float(args.map_radius),
                    sigma=float(args.map_sigma),
                    max_neighbors=int(args.map_maxn),
                    agg=str(args.map_agg),
                )

        nz = point_scores[point_scores > 0]
        vmax = float(np.percentile(nz, float(args.clip_percentile))) if nz.size > 0 else (float(point_scores.max()) if float(point_scores.max()) > 0 else 1.0)
        if vmax <= 1e-9:
            vmax = 1.0
        scores = np.clip(point_scores, 0, vmax) / (vmax + 1e-9)

        colors = colorize_points(input_xyz, scores)

        ply_path = det_dir / "heatmap.ply"
        npz_path = det_dir / "heatmap.npz"
        dbg_path = det_dir / "debug.npz"

        save_colored_ply(str(ply_path), input_xyz, colors)
        save_colored_npz(str(npz_path), input_xyz, colors, scores=scores)

        np.savez_compressed(
            dbg_path,
            sample=np.array([sample_tag], dtype=object),
            det_idx=np.array([det_idx], dtype=np.int32),
            class_name=np.array([class_name], dtype=object),
            pred_score=np.array([det_score], dtype=np.float32),
            pred_label=np.array([det_label], dtype=np.int32),
            pred_box=det_box.astype(np.float32),
            roi_idx=np.array([roi_idx_used], dtype=np.int32),
            target_logit=np.array([holder["target_logit"]], dtype=np.float32),
            pooled_frame=np.array([pooled_frame], dtype=object),
            act_layer=np.array([args.act_layer], dtype=object),
        )

        det_meta = {
            "det_idx": int(det_idx),
            "class_name": class_name,
            "score": float(det_score),
            "roi_idx": int(roi_idx_used),
            "target_logit": float(holder["target_logit"]),
            "pooled_frame": str(pooled_frame),
            "folder": str(det_dir.relative_to(sample_dir)),
            "ply": str(ply_path.relative_to(sample_dir)),
            "npz": str(npz_path.relative_to(sample_dir)),
            "dbg": str(dbg_path.relative_to(sample_dir)),
            "pred_box": det_box.tolist(),
            "pred_label": int(det_label),
        }
        detection_meta_list.append(det_meta)

        print(f"[SAVED] {class_name} det{det_idx:03d} score={det_score:.6f} roi={roi_idx_used} -> {det_dir}")

    cam.remove_hooks()

    meta = {
        "sample": sample_tag,
        "sonar_npy": str(Path(args.sonar_npy).resolve()),
        "mode": "per_detect",
        "act_layer": args.act_layer,
        "detections": detection_meta_list
    }
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for d in detection_meta_list:
            w.writerow({
                "sample": sample_tag,
                "det_idx": d["det_idx"],
                "class_name": d["class_name"],
                "score": d["score"],
                "roi_idx": d["roi_idx"],
                "target_logit": d["target_logit"],
                "pooled_frame": d["pooled_frame"],
                "folder": d["folder"],
                "npz": d["npz"],
                "ply": d["ply"],
                "dbg": d["dbg"],
            })

    print(f"[OK] Saved per-detection Grad-CAM -> {sample_dir}")
    print(f"[OK] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
