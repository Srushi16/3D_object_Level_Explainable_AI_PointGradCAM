#!/usr/bin/env python3
"""
PointGrad-CAM runner for OpenPCDet PointRCNN — PER-DETECTION batch export
UPDATED (2026-01-21): CONSISTENT evaluation metrics aligned with UPDATED LIME/Occlusion.

Outputs:
- per detection: det*/heatmap.npz (contains deletion_* arrays + metrics)
- per sample: metadata.json, detections.csv, metrics_summary.csv
Compatible with plot_xai_metrics.py.
"""

import os
import sys
import argparse
import random
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

try:
    from pcdet.datasets.kitti.kitti_zip_dataset import KittiZipDataset
except Exception:
    KittiZipDataset = None

from pointgradCAM.gradcam import PointGradCAM

try:
    from pointgradCAM.visualize import colorize_points, save_colored_ply
except Exception:
    colorize_points = None
    save_colored_ply = None

IOU_UTIL = None
try:
    from pcdet.ops.iou3d_nms import iou3d_nms_utils as IOU_UTIL  # type: ignore
except Exception:
    IOU_UTIL = None


def list_relevant_modules(model: torch.nn.Module) -> None:
    print("\n=== Relevant modules (roi/sa/fp/backbone/point_head) ===")
    for name, _ in model.named_modules():
        low = name.lower()
        if any(k in low for k in ["roi", "sa_modules", "fp_modules", "backbone_3d", "point_head"]):
            print(name)


def safe_score_tag(score: float) -> str:
    return f"{score:.3f}".replace(".", "p")


def ensure_xyzi(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Expected points shape (N,>=4), got {pts.shape}")
    return pts[:, :4].astype(np.float32, copy=False)


def pad_or_sample_to(pts_xyzi: np.ndarray, n_target: int, rng: np.random.RandomState) -> np.ndarray:
    pts_xyzi = ensure_xyzi(pts_xyzi)
    if pts_xyzi.shape[0] == 0:
        return np.zeros((1, 4), dtype=np.float32)

    n_target = int(n_target)
    if n_target <= 0:
        return pts_xyzi

    n = int(pts_xyzi.shape[0])
    if n == n_target:
        return pts_xyzi
    if n > n_target:
        idx = rng.choice(n, n_target, replace=False)
        return pts_xyzi[idx]

    idx = rng.choice(n, n_target - n, replace=True)
    return np.concatenate([pts_xyzi, pts_xyzi[idx]], axis=0)


def make_input_dict_from_points(pts_xyzi: np.ndarray, device: torch.device) -> Dict[str, Any]:
    pts_xyzi = ensure_xyzi(pts_xyzi)
    if pts_xyzi.shape[0] == 0:
        pts_xyzi = np.zeros((1, 4), dtype=np.float32)

    pts_with_batch = np.hstack(
        [np.zeros((pts_xyzi.shape[0], 1), dtype=np.float32), pts_xyzi]
    ).astype(np.float32)
    points_tensor = torch.from_numpy(pts_with_batch).to(device)
    return {"batch_size": 1, "points": points_tensor}


def best_roi_match(rois_lidar: np.ndarray, det_box_lidar: np.ndarray) -> int:
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
    Rm = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)
    return (pooled_xyz_roi @ Rm.T) + np.array([cx, cy, cz], dtype=np.float32)[None, :]


def detect_pooled_frame(xyz_roi: np.ndarray, roi_box: np.ndarray) -> str:
    center = roi_box[:3].astype(np.float32)
    mean_xyz = xyz_roi.mean(axis=0).astype(np.float32)

    dist_to_center = float(np.linalg.norm(mean_xyz - center))
    dist_to_zero = float(np.linalg.norm(mean_xyz))

    dx, dy, _ = roi_box[3:6].astype(np.float32)
    canonical_like = (dist_to_zero < 5.0) and (np.abs(mean_xyz[0]) < max(5.0, dx * 2)) and (np.abs(mean_xyz[1]) < max(5.0, dy * 2))
    world_like = dist_to_center < 3.0

    if world_like and not canonical_like:
        return "world"
    if canonical_like and not world_like:
        return "canonical"
    return "world"


def points_in_box_mask(xyz: np.ndarray, box7: np.ndarray, expand: float = 1.0) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(np.float32)
    dx *= float(expand)
    dy *= float(expand)
    dz *= float(expand)

    pts = xyz.astype(np.float32) - np.array([cx, cy, cz], dtype=np.float32)[None, :]
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    Rm = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)
    pts_local = pts @ Rm.T

    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    return (np.abs(pts_local[:, 0]) <= hx) & (np.abs(pts_local[:, 1]) <= hy) & (np.abs(pts_local[:, 2]) <= hz)


def compute_focus_ratio(abs_imp: np.ndarray, region_mask: np.ndarray) -> float:
    denom = float(abs_imp.sum())
    if denom <= 1e-12:
        return 0.0
    return float(abs_imp[region_mask].sum()) / denom


def compute_sparsity_all_points(abs_imp: np.ndarray, alpha: float) -> float:
    denom = float(abs_imp.sum())
    if denom <= 1e-12:
        return 0.0
    N = abs_imp.shape[0]
    k = int(max(1, round(alpha * N)))
    idx = np.argpartition(abs_imp, -k)[-k:]
    return float(abs_imp[idx].sum()) / denom


def run_inference_once(model: torch.nn.Module, pts_xyzi: np.ndarray, device: torch.device) -> Dict[str, Any]:
    inp = make_input_dict_from_points(pts_xyzi, device)
    with torch.no_grad():
        outputs = model(inp)
    pred_dicts = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return pred_dicts[0]


def score_same_detection_strict(
    model: torch.nn.Module,
    pts_xyzi: np.ndarray,
    det_box7: np.ndarray,
    det_label: int,
    device: torch.device,
    tau_iou: float = 0.10
) -> float:
    pts_xyzi = ensure_xyzi(pts_xyzi)
    if pts_xyzi.shape[0] == 0:
        pts_xyzi = np.zeros((1, 4), dtype=np.float32)

    pred0 = run_inference_once(model, pts_xyzi, device)
    if "pred_scores" not in pred0 or pred0["pred_scores"].numel() == 0:
        return 0.0

    boxes = pred0["pred_boxes"].detach().cpu().numpy()
    scores = pred0["pred_scores"].detach().cpu().numpy()
    labels = pred0["pred_labels"].detach().cpu().numpy()

    same = (labels == det_label)
    if not np.any(same):
        return 0.0

    boxes_f = boxes[same]
    scores_f = scores[same]

    if IOU_UTIL is None:
        return float(scores_f.max()) if scores_f.size else 0.0

    ious = IOU_UTIL.boxes_iou3d_gpu(
        torch.from_numpy(boxes_f).float().to(device),
        torch.from_numpy(det_box7[None, :]).float().to(device)
    )[0].detach().cpu().numpy().flatten()

    best = int(np.argmax(ious))
    if float(ious[best]) >= float(tau_iou):
        return float(scores_f[best])
    return 0.0


def deletion_curve_strict(
    model: torch.nn.Module,
    pts_eval: np.ndarray,
    abs_imp_eval: np.ndarray,
    det_box7: np.ndarray,
    det_label: int,
    device: torch.device,
    fracs: List[float],
    tau_iou: float,
    n_target: int,
    rng_eval: np.random.RandomState,
    baseline_eval: float,
) -> Tuple[List[float], List[float], float]:
    pts_eval = ensure_xyzi(pts_eval)
    abs_imp_eval = np.asarray(abs_imp_eval, dtype=np.float32).reshape(-1)

    N = int(abs_imp_eval.shape[0])
    if N != int(pts_eval.shape[0]):
        raise ValueError(f"abs_imp_eval length {N} != points length {pts_eval.shape[0]}")

    fracs = [float(x) for x in fracs]
    if len(fracs) == 0 or fracs[0] != 0.0:
        fracs = [0.0] + fracs

    order = np.argsort(-abs_imp_eval)

    scores_list: List[float] = []
    for f in fracs:
        f = float(f)

        if f <= 0.0:
            scores_list.append(float(baseline_eval))
            continue

        k = int(round(f * N))
        k = max(1, min(k, N))
        remove_idx = order[:k]
        keep_mask = np.ones(N, dtype=bool)
        keep_mask[remove_idx] = False
        pts_keep = pts_eval[keep_mask]

        if pts_keep.shape[0] == 0:
            pts_keep = np.zeros((1, 4), dtype=np.float32)

        pts_keep = pad_or_sample_to(pts_keep, n_target=n_target, rng=rng_eval)
        s = score_same_detection_strict(model, pts_keep, det_box7, det_label, device, tau_iou=tau_iou)
        scores_list.append(float(s))

    x = np.array(fracs, dtype=np.float32)
    y = np.array(scores_list, dtype=np.float32)
    auc = float(np.trapz(y, x))
    return fracs, scores_list, auc


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
    radius: float = 0.40,
    sigma: float = 0.18,
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


def main():
    parser = argparse.ArgumentParser("PointGrad-CAM (per-detection) for PointRCNN (OpenPCDet) + CONSISTENT metrics")

    parser.add_argument("--cfg", required=True)    # TODO(PATH): model config YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    parser.add_argument("--ckpt", required=True)   # TODO(PATH): checkpoint path (e.g., checkpoints/pointrcnn_7870.pth)
    parser.add_argument("--kitti", required=True)  # TODO(PATH): KITTI root (velodyne/ or training/velodyne/ must exist)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--outdir", default="./gradcam_outputs")  # TODO(PATH): output folder
    parser.add_argument("--list-modules", action="store_true")

    parser.add_argument("--act-layer", default="roi_head.merge_down_layer")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--min_score", type=float, default=0.10)
    parser.add_argument("--min_points_in_box", type=int, default=10)

    parser.add_argument("--map-mode", default="radius", choices=["radius", "knn"])
    parser.add_argument("--map-k", type=int, default=20)
    parser.add_argument("--map-radius", type=float, default=0.50)
    parser.add_argument("--map-sigma", type=float, default=0.22)
    parser.add_argument("--map-maxn", type=int, default=256)
    parser.add_argument("--map-agg", default="sum", choices=["sum", "max"])

    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--raw-bin", action="store_true")

    parser.add_argument("--metrics_alpha", type=float, default=0.05)
    parser.add_argument("--deletion_fracs", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--focus_expand", type=float, default=1.6)
    parser.add_argument("--tau_iou", type=float, default=0.10)

    parser.add_argument("--n_target", type=int, default=16384)
    parser.add_argument("--fixed_eval_subset", action="store_true", default=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_from_yaml_file(args.cfg, cfg)

    class DummyDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False)
            self.point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            raise NotImplementedError

    dummy_dataset = DummyDataset()
    logger = common_utils.create_logger()

    print(f"Building model from {args.cfg} on {device} ...")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.to(device)
    model.eval()

    if args.list_modules:
        list_relevant_modules(model)
        return

    torch.set_grad_enabled(True)

    print(f"Loading points for sample {args.sample:06d} ...")
    if args.raw_bin or KittiZipDataset is None:
        p1 = os.path.join(args.kitti, "velodyne", f"{args.sample:06d}.bin")
        p2 = os.path.join(args.kitti, "training", "velodyne", f"{args.sample:06d}.bin")
        bin_path = p1 if os.path.exists(p1) else p2
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Could not find velodyne bin at {p1} or {p2}")
        pts_np = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    else:
        dataset = KittiZipDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(args.kitti),
            logger=logger,
        )
        data_dict = dataset[args.sample]
        pts_np = np.asarray(data_dict["points"], dtype=np.float32)

    pts_xyzi_raw = ensure_xyzi(pts_np)

    n_target = int(args.n_target)
    rng_sample = np.random.RandomState(int(SEED) * 100000 + int(args.sample))

    if args.fixed_eval_subset and n_target > 0 and pts_xyzi_raw.shape[0] > n_target:
        eval_idx0 = rng_sample.choice(pts_xyzi_raw.shape[0], n_target, replace=False)
        pts_eval0 = pts_xyzi_raw[eval_idx0]
    else:
        eval_idx0 = np.arange(pts_xyzi_raw.shape[0], dtype=np.int64)
        pts_eval0 = pts_xyzi_raw

    pts_eval0 = pad_or_sample_to(pts_eval0, n_target=n_target, rng=rng_sample)
    input_xyz = pts_eval0[:, :3]

    capture: Dict[str, Any] = {"rois": None, "pooled_xyz": None, "rcnn_logits": None}

    roipool_name = "roi_head.roipoint_pool3d_layer"
    roipool_mod = dict(model.named_modules()).get(roipool_name, None)
    if roipool_mod is not None:
        def roipool_hook(m, inp, out):
            try:
                rois = inp[2]
                if torch.is_tensor(rois) and rois.dim() == 3 and rois.shape[-1] == 7:
                    capture["rois"] = rois
            except Exception:
                pass

            try:
                if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                    pooled = out[0]
                    if pooled.dim() == 4 and pooled.shape[-1] >= 3:
                        capture["pooled_xyz"] = pooled[..., 0:3].squeeze(0).contiguous()
            except Exception:
                pass

        roipool_mod.register_forward_hook(roipool_hook)

    cls7_name = "roi_head.cls_layers.7"
    cls7_mod = dict(model.named_modules()).get(cls7_name, None)
    if cls7_mod is None:
        raise RuntimeError(f"{cls7_name} not found.")
    else:
        def cls7_hook(m, inp, out):
            capture["rcnn_logits"] = out
        cls7_mod.register_forward_hook(cls7_hook)

    pred0 = run_inference_once(model, pts_eval0, device)
    if "pred_boxes" not in pred0 or pred0["pred_boxes"].numel() == 0:
        print("[INFO] No detections in this sample.")
        return

    boxes_all = pred0["pred_boxes"].detach().cpu().numpy()
    scores_all = pred0["pred_scores"].detach().cpu().numpy()
    labels_all = pred0["pred_labels"].detach().cpu().numpy()

    sample_dir = Path(args.outdir) / f"sample_{args.sample:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    per_class_dir: Dict[str, Path] = {cls: (sample_dir / cls) for cls in cfg.CLASS_NAMES}
    for p in per_class_dir.values():
        p.mkdir(parents=True, exist_ok=True)

    det_candidates: List[int] = []
    objects_per_class = {cls: 0 for cls in cfg.CLASS_NAMES}
    points_in_box_counts: Dict[int, int] = {}

    for det_idx in range(len(boxes_all)):
        score = float(scores_all[det_idx])
        label = int(labels_all[det_idx])
        class_name = cfg.CLASS_NAMES[label - 1] if 1 <= label <= len(cfg.CLASS_NAMES) else f"cls{label}"

        if score < float(args.min_score):
            continue

        m = points_in_box_mask(input_xyz, boxes_all[det_idx][:7], expand=1.0)
        n_in = int(m.sum())
        points_in_box_counts[det_idx] = n_in

        if n_in < int(args.min_points_in_box):
            continue

        det_candidates.append(det_idx)
        if class_name in objects_per_class:
            objects_per_class[class_name] += 1

    if len(det_candidates) == 0:
        meta = {
            "sample_idx": int(args.sample),
            "num_points_raw": int(pts_xyzi_raw.shape[0]),
            "num_points_eval": int(pts_eval0.shape[0]),
            "eval_idx0": eval_idx0.tolist(),
            "objects_per_class": objects_per_class,
            "detections": [],
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        return

    cam = PointGradCAM(model, args.act_layer, device=device)
    cam.verbose = args.verbose

    detection_meta_list: List[Dict[str, Any]] = []

    detections_csv = sample_dir / "detections.csv"
    detections_fields = [
        "sample_idx", "det_idx", "class_name", "score",
        "points_in_box", "roi_idx", "target_logit", "pooled_frame",
        "map_mode", "map_k", "map_radius", "map_sigma", "map_maxn", "map_agg",
        "clip_percentile",
        "metrics_alpha", "focus_expand", "tau_iou", "n_target", "fixed_eval_subset",
        "baseline_eval", "focus_ratio", "sparsity", "delta_s_alpha", "deletion_auc",
        "folder", "npz", "ply", "dbg"
    ]

    metrics_csv = sample_dir / "metrics_summary.csv"
    metrics_fields = [
        "sample_idx", "det_idx", "class_name", "pred_score",
        "metrics_alpha", "focus_expand", "tau_iou",
        "baseline_eval", "focus_ratio", "sparsity", "delta_s_alpha", "deletion_auc",
        "npz_path"
    ]

    for det_idx in det_candidates:
        det_box = boxes_all[det_idx][:7].astype(np.float32)
        det_score = float(scores_all[det_idx])
        det_label = int(labels_all[det_idx])
        class_name = cfg.CLASS_NAMES[det_label - 1] if 1 <= det_label <= len(cfg.CLASS_NAMES) else f"cls{det_label}"
        n_in = int(points_in_box_counts.get(det_idx, 0))

        det_dir = per_class_dir.get(class_name, sample_dir / class_name) / f"det{det_idx:03d}_score{safe_score_tag(det_score)}"
        det_dir.mkdir(parents=True, exist_ok=True)

        rng_det = np.random.RandomState(int(SEED) * 100000 + int(args.sample) * 1000 + int(det_idx))
        holder = {"roi_idx": 0, "target_logit": 0.0, "pooled_frame": "unknown"}

        def target_fn(outputs):
            pred_dicts = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred0_local = pred_dicts[0]

            if capture["rcnn_logits"] is None:
                raise RuntimeError("rcnn_logits not captured (cls_layers.7 hook did not fire).")

            rcnn_logits = capture["rcnn_logits"]
            if rcnn_logits.dim() == 3 and rcnn_logits.shape[-1] == 1:
                rcnn_logits = rcnn_logits.squeeze(-1)
            if rcnn_logits.dim() == 3 and rcnn_logits.shape[1] == 1:
                rcnn_logits = rcnn_logits.squeeze(1)
            if rcnn_logits.dim() != 2:
                raise RuntimeError(f"Unexpected rcnn_logits shape after squeeze: {tuple(rcnn_logits.shape)}")

            if capture["rois"] is not None and "pred_boxes" in pred0_local and pred0_local["pred_boxes"].numel() > 0:
                rois_np = capture["rois"].detach().cpu().numpy().squeeze(0)
                roi_idx = best_roi_match(rois_np, det_box)
            else:
                roi_idx = int(torch.argmax(rcnn_logits[:, 0]).item())

            holder["roi_idx"] = int(roi_idx)
            t = rcnn_logits[roi_idx, 0] if rcnn_logits.shape[1] == 1 else rcnn_logits[roi_idx].max()
            holder["target_logit"] = float(t.detach().cpu().item())
            return t

        cam_list = cam.attribute(make_input_dict_from_points(pts_eval0, device), target_fn=target_fn, retain_graph=False)
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
                point_scores = scatter_pooled_xyz_to_input_knn(
                    input_xyz, xyz_roi_world, pooled_cam_use, k=max(1, args.map_k)
                )
            else:
                point_scores = scatter_pooled_xyz_to_input_radius(
                    input_xyz, xyz_roi_world, pooled_cam_use,
                    radius=float(args.map_radius),
                    sigma=float(args.map_sigma),
                    max_neighbors=int(args.map_maxn),
                    agg=str(args.map_agg),
                )

        nonzero_vals = point_scores[point_scores > 0]
        vmax = float(np.percentile(nonzero_vals, float(args.clip_percentile))) if nonzero_vals.size > 0 else 1.0
        if vmax <= 1e-9:
            vmax = float(point_scores.max()) if float(point_scores.max()) > 0 else 1.0

        scores01 = np.clip(point_scores, 0, vmax) / (vmax + 1e-9)
        scores01 = scores01.astype(np.float32)
        scores01[scores01 < 1e-6] = 0.0
        abs_imp_eval = scores01

        region_mask = points_in_box_mask(input_xyz, det_box[:7], expand=float(args.focus_expand))
        focus_ratio = compute_focus_ratio(abs_imp_eval, region_mask)
        sparsity = compute_sparsity_all_points(abs_imp_eval, alpha=float(args.metrics_alpha))

        baseline_eval = float(det_score)

        Ne = int(abs_imp_eval.shape[0])
        k = int(max(1, round(float(args.metrics_alpha) * Ne)))
        top_idx = np.argpartition(abs_imp_eval, -k)[-k:]

        keep_mask = np.ones(Ne, dtype=bool)
        keep_mask[top_idx] = False

        pts_del = pts_eval0[keep_mask]
        if pts_del.shape[0] == 0:
            pts_del = np.zeros((1, 4), dtype=np.float32)

        pts_del = pad_or_sample_to(pts_del, n_target=n_target, rng=rng_det)

        s_del = float(score_same_detection_strict(
            model=model,
            pts_xyzi=pts_del,
            det_box7=det_box[:7],
            det_label=det_label,
            device=device,
            tau_iou=float(args.tau_iou),
        ))
        delta_s_alpha = float(baseline_eval - s_del)

        del_fracs, del_scores, del_auc = deletion_curve_strict(
            model=model,
            pts_eval=pts_eval0,
            abs_imp_eval=abs_imp_eval,
            det_box7=det_box[:7],
            det_label=det_label,
            device=device,
            fracs=[float(x) for x in args.deletion_fracs],
            tau_iou=float(args.tau_iou),
            n_target=n_target,
            rng_eval=rng_det,
            baseline_eval=baseline_eval,
        )

        ply_path = det_dir / "heatmap.ply"
        npz_path = det_dir / "heatmap.npz"
        dbg_path = det_dir / "debug.npz"

        colors = None
        if colorize_points is not None:
            colors = colorize_points(input_xyz, scores01)

        if (save_colored_ply is not None) and (colors is not None):
            try:
                save_colored_ply(str(ply_path), input_xyz, colors)
            except Exception:
                pass

        np.savez_compressed(
            npz_path,
            points=input_xyz.astype(np.float32),
            scores=scores01.astype(np.float32),
            colors=(colors.astype(np.float32) if colors is not None else np.zeros((input_xyz.shape[0], 3), dtype=np.float32)),
            metrics_alpha=np.float32(args.metrics_alpha),
            focus_expand=np.float32(args.focus_expand),
            tau_iou=np.float32(args.tau_iou),
            n_target=np.int32(n_target),
            fixed_eval_subset=np.int32(1 if args.fixed_eval_subset else 0),
            eval_idx0=np.array(eval_idx0, dtype=np.int64),
            baseline_eval=np.float32(baseline_eval),
            focus_ratio=np.float32(focus_ratio),
            sparsity=np.float32(sparsity),
            delta_s_alpha=np.float32(delta_s_alpha),
            deletion_fracs=np.array(del_fracs, dtype=np.float32),
            deletion_scores=np.array(del_scores, dtype=np.float32),
            deletion_auc=np.float32(del_auc),
            sample_idx=np.int32(args.sample),
            det_idx=np.int32(det_idx),
            class_name=np.array([class_name], dtype=object),
            pred_score=np.float32(det_score),
            pred_label=np.int32(det_label),
            pred_box=det_box.astype(np.float32),
        )

        np.savez_compressed(
            dbg_path,
            roi_idx=np.array([roi_idx_used], dtype=np.int32),
            target_logit=np.array([holder["target_logit"]], dtype=np.float32),
            pooled_cam=pooled_cam.astype(np.float32),
            pooled_frame=np.array([pooled_frame], dtype=object),
            act_layer=np.array([args.act_layer], dtype=object),
            map_mode=np.array([args.map_mode], dtype=object),
            map_k=np.array([args.map_k], dtype=np.int32),
            map_radius=np.array([args.map_radius], dtype=np.float32),
            map_sigma=np.array([args.map_sigma], dtype=np.float32),
            map_maxn=np.array([args.map_maxn], dtype=np.int32),
            map_agg=np.array([args.map_agg], dtype=object),
            clip_percentile=np.array([args.clip_percentile], dtype=np.float32),
        )

        det_meta = {
            "det_idx": int(det_idx),
            "class_name": class_name,
            "score": float(det_score),
            "points_in_box": int(n_in),
            "roi_idx": int(roi_idx_used),
            "target_logit": float(holder["target_logit"]),
            "pooled_frame": str(pooled_frame),
            "folder": str(det_dir.relative_to(sample_dir)),
            "ply": str(ply_path.relative_to(sample_dir)),
            "npz": str(npz_path.relative_to(sample_dir)),
            "dbg": str(dbg_path.relative_to(sample_dir)),
            "metrics": {
                "alpha": float(args.metrics_alpha),
                "focus_expand": float(args.focus_expand),
                "tau_iou": float(args.tau_iou),
                "n_target": int(n_target),
                "fixed_eval_subset": bool(args.fixed_eval_subset),
                "baseline_eval": float(baseline_eval),
                "focus_ratio": float(focus_ratio),
                "sparsity": float(sparsity),
                "delta_s_alpha": float(delta_s_alpha),
                "deletion_auc": float(del_auc),
            }
        }
        detection_meta_list.append(det_meta)

    cam.remove_hooks()

    meta = {
        "sample_idx": int(args.sample),
        "num_points_raw": int(pts_xyzi_raw.shape[0]),
        "num_points_eval": int(pts_eval0.shape[0]),
        "eval_idx0": eval_idx0.tolist(),
        "objects_per_class": objects_per_class,
        "filters": {
            "min_score": float(args.min_score),
            "min_points_in_box": int(args.min_points_in_box)
        },
        "metrics": {
            "alpha": float(args.metrics_alpha),
            "focus_expand": float(args.focus_expand),
            "tau_iou": float(args.tau_iou),
            "deletion_fracs": [float(x) for x in args.deletion_fracs],
            "n_target": int(args.n_target),
            "fixed_eval_subset": bool(args.fixed_eval_subset),
            "note_auc": "AUC is remaining score vs fraction removed; LOWER is better."
        },
        "detections": detection_meta_list
    }
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(detections_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detections_fields)
        w.writeheader()
        for d in detection_meta_list:
            m = d.get("metrics", {})
            w.writerow({
                "sample_idx": int(args.sample),
                "det_idx": d["det_idx"],
                "class_name": d["class_name"],
                "score": d["score"],
                "points_in_box": d["points_in_box"],
                "roi_idx": d["roi_idx"],
                "target_logit": d["target_logit"],
                "pooled_frame": d["pooled_frame"],
                "map_mode": args.map_mode,
                "map_k": int(args.map_k),
                "map_radius": float(args.map_radius),
                "map_sigma": float(args.map_sigma),
                "map_maxn": int(args.map_maxn),
                "map_agg": str(args.map_agg),
                "clip_percentile": float(args.clip_percentile),
                "metrics_alpha": m.get("alpha", args.metrics_alpha),
                "focus_expand": m.get("focus_expand", args.focus_expand),
                "tau_iou": m.get("tau_iou", args.tau_iou),
                "n_target": m.get("n_target", args.n_target),
                "fixed_eval_subset": int(1 if m.get("fixed_eval_subset", False) else 0),
                "baseline_eval": m.get("baseline_eval", None),
                "focus_ratio": m.get("focus_ratio", None),
                "sparsity": m.get("sparsity", None),
                "delta_s_alpha": m.get("delta_s_alpha", None),
                "deletion_auc": m.get("deletion_auc", None),
                "folder": d["folder"],
                "npz": d["npz"],
                "ply": d["ply"],
                "dbg": d["dbg"],
            })

    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metrics_fields)
        w.writeheader()
        for d in detection_meta_list:
            m = d.get("metrics", {})
            w.writerow({
                "sample_idx": int(args.sample),
                "det_idx": d["det_idx"],
                "class_name": d["class_name"],
                "pred_score": d["score"],
                "metrics_alpha": m.get("alpha", args.metrics_alpha),
                "focus_expand": m.get("focus_expand", args.focus_expand),
                "tau_iou": m.get("tau_iou", args.tau_iou),
                "baseline_eval": m.get("baseline_eval", None),
                "focus_ratio": m.get("focus_ratio", None),
                "sparsity": m.get("sparsity", None),
                "delta_s_alpha": m.get("delta_s_alpha", None),
                "deletion_auc": m.get("deletion_auc", None),
                "npz_path": str((sample_dir / d["npz"]).resolve()),
            })

    print(f"\nSaved sample metadata : {sample_dir / 'metadata.json'}")
    print(f"Saved detections CSV  : {detections_csv}")
    print(f"Saved metrics summary : {metrics_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
