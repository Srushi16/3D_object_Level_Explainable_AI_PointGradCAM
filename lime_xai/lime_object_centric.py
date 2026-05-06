#!/usr/bin/env python3
"""
lime_object_centric.py — PointRCNN LIME (KITTI) with TRUE object-centric mode
UPDATED (2026-01-21): CONSISTENT evaluation metrics across LIME/Occlusion/PointGradCAM.

Outputs (plot_xai_metrics.py compatible):
- Per-detection NPZ includes deletion_fracs, deletion_scores, deletion_auc
- sample metrics_summary.csv includes npz (relative), folder, sample_dir
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import logging

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_zip_dataset import KittiZipDataset, kitti_zip_dataset_collate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

FIXED_NUM_POINTS = 16384


def setup_logger():
    logger = logging.getLogger("LIME3D_OBJECT_CENTRIC")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger


logger = setup_logger()


def fps_init_centers(xyz: np.ndarray, n: int) -> np.ndarray:
    """FPS init centers using PointNet2 op (good KMeans init)."""
    import torch as _torch
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils

    if xyz.shape[0] == 0 or n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = _torch.from_numpy(xyz).float().cuda().unsqueeze(0)  # (1,N,3)
    idx_tensor, = pointnet2_utils.FarthestPointSampling.apply(
        pts, _torch.tensor([n], device=pts.device))
    idx = idx_tensor[0]
    centers = pts[0][idx].cpu().numpy()
    return centers.reshape(-1, 3) if centers.ndim == 1 else centers


def strip_gt_fields(d: dict) -> dict:
    """Remove GT so model sees test-time inputs only."""
    return {k: v for k, v in d.items()
            if k not in ('gt_boxes', 'gt_names', 'gt_boxes2d')}


def points_in_box_mask(points_xyz: np.ndarray, box7: np.ndarray, expand: float = 1.0) -> np.ndarray:
    """
    points_xyz: (N,3)
    box7: [cx, cy, cz, l, w, h, yaw]
    returns: (N,) bool for inside expanded oriented box
    """
    cx, cy, cz, l, w, h, yaw = box7.astype(np.float32)
    l *= float(expand)
    w *= float(expand)
    h *= float(expand)

    p = points_xyz.astype(np.float32) - np.array([cx, cy, cz], dtype=np.float32)

    c = np.cos(-yaw).astype(np.float32)
    s = np.sin(-yaw).astype(np.float32)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    pr = p @ R.T

    return ((np.abs(pr[:, 0]) <= l / 2) &
            (np.abs(pr[:, 1]) <= w / 2) &
            (np.abs(pr[:, 2]) <= h / 2))


def iou3d_match(target: np.ndarray, pred_boxes: np.ndarray, thresh: float = 0.0):
    """Return (best_idx, best_iou) or (-1, best_iou) if below thresh."""
    if pred_boxes.shape[0] == 0:
        return -1, 0.0

    tgt = torch.from_numpy(target[None]).float().cuda()
    prd = torch.from_numpy(pred_boxes).float().cuda()
    with torch.no_grad():
        iou = iou3d_nms_utils.boxes_iou3d_gpu(prd, tgt).cpu().numpy().flatten()

    best = int(np.argmax(iou))
    best_iou = float(iou[best])
    return (best, best_iou) if best_iou >= thresh else (-1, best_iou)


def ensure_xyzi(pts: np.ndarray) -> np.ndarray:
    """Ensure (N,4) float32 points."""
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Expected points shape (N,>=4), got {pts.shape}")
    return pts[:, :4].astype(np.float32, copy=False)


def safe_build_pert_object_centric(outside_pts: np.ndarray,
                                  inside_cluster_pts: list,
                                  keep_ids: np.ndarray,
                                  rng: np.random.RandomState) -> np.ndarray:
    """Keep outside fixed; vary kept inside clusters; pad/downsample to FIXED_NUM_POINTS (deterministic)."""
    kept_inside = [inside_cluster_pts[i] for i in keep_ids
                   if 0 <= i < len(inside_cluster_pts) and inside_cluster_pts[i].size > 0]
    if not kept_inside and len(inside_cluster_pts) > 0:
        kept_inside = [inside_cluster_pts[0]]

    if kept_inside:
        pts = np.vstack([outside_pts] + kept_inside).astype(np.float32)
    else:
        pts = outside_pts.astype(np.float32)

    N = FIXED_NUM_POINTS
    if pts.shape[0] == 0:
        return np.zeros((N, 4), dtype=np.float32)

    if pts.shape[0] < N:
        extra = pts[rng.choice(pts.shape[0], N - pts.shape[0], replace=True)]
        pts = np.vstack([pts, extra])
    elif pts.shape[0] > N:
        pts = pts[rng.choice(pts.shape[0], N, replace=False)]

    return pts.astype(np.float32, copy=False)


def safe_build_pert_global(cluster_pts: list, keep_ids: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Global perturbation: keep only selected clusters; pad/downsample to FIXED_NUM_POINTS (deterministic)."""
    pts_list = [cluster_pts[i] for i in keep_ids
                if 0 <= i < len(cluster_pts) and cluster_pts[i].size > 0]
    if not pts_list and len(cluster_pts) > 0:
        pts_list = [cluster_pts[0]]

    pts = np.vstack(pts_list).astype(np.float32)
    N = FIXED_NUM_POINTS

    if pts.shape[0] < N:
        extra = pts[rng.choice(pts.shape[0], N - pts.shape[0], replace=True)]
        pts = np.vstack([pts, extra])
    elif pts.shape[0] > N:
        pts = pts[rng.choice(pts.shape[0], N, replace=False)]
    return pts.astype(np.float32, copy=False)


def collate_and_pad(batch_pts, max_points=FIXED_NUM_POINTS):
    """List of (N,C) -> batch dict for PointRCNN."""
    padded, cnt = [], []
    for pts in batch_pts:
        pts = ensure_xyzi(pts)
        N = pts.shape[0]
        cnt.append(min(N, max_points))
        if N >= max_points:
            padded.append(pts[:max_points])
        else:
            pad = np.zeros((max_points - N, pts.shape[1]), dtype=np.float32)
            padded.append(np.vstack([pts, pad]))

    batch_tensor = torch.tensor(np.stack(padded), dtype=torch.float32).cuda()
    xyz_batch_cnt = torch.tensor(cnt, dtype=torch.int32).cuda()
    return {'points': batch_tensor, 'xyz_batch_cnt': xyz_batch_cnt}


def run_model_batch(model, batch):
    """Run PointRCNN on batch (adds batch indices)."""
    load_data_to_gpu(batch)
    B, N, C = batch['points'].shape
    batch['batch_size'] = B
    batch['xyz_batch_cnt'] = batch['xyz_batch_cnt'].to(dtype=torch.int32, device=batch['points'].device)

    batch_idx = torch.arange(B, device=batch['points'].device).view(B, 1, 1).repeat(1, N, 1)
    points_with_idx = torch.cat([batch_idx.float(), batch['points']], dim=-1).view(-1, C + 1)
    batch['points'] = points_with_idx

    with torch.no_grad():
        pred_dicts, _ = model(batch)
    return pred_dicts


def choose_expand_base(class_name: str) -> float:
    cls = class_name.lower()
    if "ped" in cls:
        return 1.8
    if "cycl" in cls:
        return 1.6
    return 1.3


def choose_expand_adaptive(class_name: str,
                           inside_count_at_base: int,
                           base_expand: float,
                           expand_min: float,
                           expand_max: float) -> float:
    e = float(base_expand)
    if inside_count_at_base < 60:
        e += 0.4
    elif inside_count_at_base < 100:
        e += 0.2
    elif inside_count_at_base > 600:
        e -= 0.1

    e = max(e, base_expand)
    e = float(np.clip(e, expand_min, expand_max))
    return e


def choose_num_clusters(class_name: str,
                        inside_count: int,
                        max_clusters: int,
                        target_pts_per_cluster: int,
                        kmin: int) -> int:
    cls = class_name.lower()
    if "ped" in cls:
        class_cap = 48
    elif "cycl" in cls:
        class_cap = 96
    else:
        class_cap = 256

    cap = int(min(max_clusters, class_cap))
    cap = max(cap, kmin)

    if inside_count <= 0:
        return kmin

    K = int(round(inside_count / float(target_pts_per_cluster)))
    K = int(np.clip(K, kmin, cap))
    K = min(K, inside_count)
    K = max(K, 1)
    return K


def compute_focus_ratio(abs_importance: np.ndarray, region_mask: np.ndarray) -> float:
    denom = float(abs_importance.sum())
    if denom <= 1e-12 or region_mask is None:
        return 0.0
    num = float(abs_importance[region_mask].sum())
    return num / denom


def compute_sparsity_all_points(abs_importance: np.ndarray, alpha: float) -> float:
    """Sparsity@alpha computed on ALL points."""
    denom = float(abs_importance.sum())
    if denom <= 1e-12:
        return 0.0
    N = abs_importance.shape[0]
    k = int(max(1, round(alpha * N)))
    idx = np.argpartition(abs_importance, -k)[-k:]
    num = float(abs_importance[idx].sum())
    return num / denom


def pad_to_fixed(pts_xyzi: np.ndarray, N: int, rng: np.random.RandomState) -> np.ndarray:
    """Deterministic pad/downsample to N points."""
    pts_xyzi = ensure_xyzi(pts_xyzi)
    if pts_xyzi.shape[0] == 0:
        return np.zeros((N, 4), dtype=np.float32)

    if pts_xyzi.shape[0] < N:
        extra = pts_xyzi[rng.choice(pts_xyzi.shape[0], N - pts_xyzi.shape[0], replace=True)]
        return np.vstack([pts_xyzi, extra]).astype(np.float32)
    if pts_xyzi.shape[0] > N:
        return pts_xyzi[rng.choice(pts_xyzi.shape[0], N, replace=False)].astype(np.float32)
    return pts_xyzi.astype(np.float32)


def score_same_detection_strict(model,
                                pts_fixed_xyzi: np.ndarray,
                                det_box7: np.ndarray,
                                det_label: int,
                                iou_thresh: float) -> float:
    """STRICT match score (0 if not matched)."""
    pts_fixed_xyzi = ensure_xyzi(pts_fixed_xyzi)
    batch_dict = collate_and_pad([pts_fixed_xyzi])
    pred = run_model_batch(model, batch_dict)[0]

    if pred['pred_boxes'].shape[0] == 0:
        return 0.0

    pboxes = pred['pred_boxes'].detach().cpu().numpy()
    pscores = pred['pred_scores'].detach().cpu().numpy()
    plabels = pred['pred_labels'].detach().cpu().numpy()

    same_class = plabels == det_label
    if not np.any(same_class):
        return 0.0

    pboxes_f = pboxes[same_class]
    pscores_f = pscores[same_class]

    match_idx, _ = iou3d_match(det_box7, pboxes_f, thresh=iou_thresh)
    if match_idx != -1:
        return float(pscores_f[match_idx])

    return 0.0


def deletion_curve_strict(model,
                          points_eval_xyzi: np.ndarray,
                          abs_importance_eval: np.ndarray,
                          det_box7: np.ndarray,
                          det_label: int,
                          fracs: list,
                          iou_thresh: float,
                          rng: np.random.RandomState,
                          batch_limit: int = 16):
    """Deletion curve on a fixed eval set (strict match scoring)."""
    points_eval_xyzi = ensure_xyzi(points_eval_xyzi)
    abs_importance_eval = np.asarray(abs_importance_eval, dtype=np.float32)

    N = int(abs_importance_eval.shape[0])
    if N != int(points_eval_xyzi.shape[0]):
        raise ValueError(f"abs_importance_eval length {N} != points_eval length {points_eval_xyzi.shape[0]}")

    order = np.argsort(-abs_importance_eval)
    fracs_sorted = sorted([float(f) for f in fracs])
    if len(fracs_sorted) == 0 or fracs_sorted[0] != 0.0:
        fracs_sorted = [0.0] + fracs_sorted

    scores = []
    for start in range(0, len(fracs_sorted), batch_limit):
        sub = fracs_sorted[start:start + batch_limit]
        clouds = []
        for f in sub:
            k = int(round(float(f) * N))
            if k <= 0:
                pts_keep = points_eval_xyzi
            else:
                remove_idx = order[:k]
                keep_mask = np.ones(N, dtype=bool)
                keep_mask[remove_idx] = False
                pts_keep = points_eval_xyzi[keep_mask]
                if pts_keep.shape[0] == 0:
                    pts_keep = np.zeros((1, 4), dtype=np.float32)

            clouds.append(pad_to_fixed(pts_keep, FIXED_NUM_POINTS, rng=rng))

        batch_dict = collate_and_pad(clouds)
        preds = run_model_batch(model, batch_dict)

        for j, _f in enumerate(sub):
            pred = preds[j]
            if pred['pred_boxes'].shape[0] == 0:
                scores.append(0.0)
                continue

            pboxes = pred['pred_boxes'].detach().cpu().numpy()
            pscores = pred['pred_scores'].detach().cpu().numpy()
            plabels = pred['pred_labels'].detach().cpu().numpy()

            same_class = plabels == det_label
            if not np.any(same_class):
                scores.append(0.0)
                continue

            pboxes_f = pboxes[same_class]
            pscores_f = pscores[same_class]
            match_idx, _ = iou3d_match(det_box7, pboxes_f, thresh=iou_thresh)
            scores.append(float(pscores_f[match_idx]) if match_idx != -1 else 0.0)

    x = np.array(fracs_sorted, dtype=np.float32)
    y = np.array(scores, dtype=np.float32)
    auc = float(np.trapz(y, x))
    return scores, auc, fracs_sorted


def save_metadata_json(sample_idx, points, detections, class_names, out_dir, object_centric, filters):
    objects_per_class = {cls: 0 for cls in class_names}
    for det in detections:
        objects_per_class[det['class_name']] += 1

    metadata = {
        "sample_idx": int(sample_idx),
        "num_points": int(points.shape[0]),
        "object_centric": bool(object_centric),
        "filters": filters,
        "objects_per_class": objects_per_class,
        "detections": detections
    }

    json_path = Path(out_dir) / f"sample_{sample_idx:06d}_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f" → Metadata JSON saved: {json_path}")
    return json_path


def save_metrics_csv(out_dir: Path, detections: list):
    """Write metrics_summary.csv (includes columns used by plot_xai_metrics.py)."""
    csv_path = out_dir / "metrics_summary.csv"
    fieldnames = [
        "det_idx", "class_name", "score",
        "baseline_eval",
        "object_centric", "expand_used", "inside_points", "num_clusters",
        "focus_ratio", "sparsity_alpha", "sparsity",
        "delta_s_alpha", "deletion_auc",
        "r2", "fallback_reason",
        "npz", "folder", "sample_dir",
        "matched_gt_idx", "matched_gt_iou",
        "eval_size"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in detections:
            row = {k: d.get(k, None) for k in fieldnames}
            row["sample_dir"] = str(out_dir)
            w.writerow(row)

    logger.info(f" → Metrics CSV saved: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser("Object-centric LIME for PointRCNN (KITTI) + CONSISTENT metrics")

    parser.add_argument('--cfg', required=True)         # TODO(PATH): model config YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    parser.add_argument('--ckpt', required=True)        # TODO(PATH): checkpoint file path (e.g., checkpoints/pointrcnn.pth)
    parser.add_argument('--root_path', required=True)   # TODO(PATH): KITTI root directory
    parser.add_argument('--info_dir', required=True)    # TODO(PATH): directory containing kitti_infos_*.pkl (and dbinfos)
    parser.add_argument('--sample_idx', type=int, default=0)

    parser.add_argument('--num_clusters', type=int, default=256,
                        help="Maximum clusters (adaptive policy chooses <= this)")
    parser.add_argument('--num_perturbs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--min_keep_frac', type=float, default=0.4)
    parser.add_argument('--tau_iou', type=float, default=0.10,
                        help="STRICT IoU threshold for same-object matching (KITTI recommended: 0.10)")
    parser.add_argument('--min_score', type=float, default=0.0)

    parser.add_argument('--object_centric', action='store_true',
                        help="Use object-centric clustering/perturbation; fallback to global if too sparse.")
    parser.add_argument('--min_inside_pts', type=int, default=50,
                        help="If inside points after adaptive expansion below this -> fallback to GLOBAL.")
    parser.add_argument('--box_expand', type=float, default=1.0,
                        help="Global lower-bound modifier for base expansion (kept for compatibility).")
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help="Enable adaptive per-detection expand + cluster selection (recommended).")
    parser.add_argument('--expand_min', type=float, default=1.2)
    parser.add_argument('--expand_max', type=float, default=2.4)
    parser.add_argument('--target_ppc', type=int, default=12,
                        help="Target points-per-cluster (smaller => more clusters).")
    parser.add_argument('--kmin', type=int, default=8,
                        help="Minimum clusters (if enough points).")

    parser.add_argument('--compute_metrics', action='store_true', default=True)
    parser.add_argument('--metrics_alpha', type=float, default=0.05,
                        help="Alpha for Sparsity and ΔScore@alpha. Default 0.05 (5%).")
    parser.add_argument('--focus_expand', type=float, default=1.6,
                        help="Focus ratio region expansion factor (KITTI recommended: 1.6).")
    parser.add_argument('--deletion_fracs', type=str, default="0,0.01,0.05,0.1,0.2,0.3,0.5",
                        help="Comma-separated fractions for deletion curve (0..1).")
    parser.add_argument('--deletion_batch', type=int, default=8,
                        help="Batch size for deletion curve evaluation per detection.")

    parser.add_argument('--out', required=True,
                        help="Any filename inside your results directory (parent dir is used). Example: /path/results/dummy.npz")  # TODO(PATH): choose your results folder
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    RNG = np.random.RandomState(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    del_fracs = [float(x.strip()) for x in args.deletion_fracs.split(",") if x.strip() != ""]
    del_fracs = sorted([max(0.0, min(1.0, f)) for f in del_fracs])

    cfg_from_yaml_file(args.cfg, cfg)
    cfg.DATA_CONFIG.INFO_DIR = str(Path(args.info_dir).absolute())

    dataset = KittiZipDataset(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
        root_path=Path(args.root_path),
        logger=common_utils.create_logger())
    dataset.set_split('val')

    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=args.ckpt, logger=common_utils.create_logger())
    model.cuda().eval()

    data_dict = dataset[args.sample_idx]
    base_dict = strip_gt_fields(data_dict)
    points = ensure_xyzi(base_dict['points'])
    num_points = points.shape[0]

    batch = kitti_zip_dataset_collate([base_dict])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    orig_pred = pred_dicts[0]

    boxes = orig_pred['pred_boxes'].detach().cpu().numpy()
    scores = orig_pred['pred_scores'].detach().cpu().numpy()
    labels = orig_pred['pred_labels'].detach().cpu().numpy()
    gt_boxes = data_dict.get('gt_boxes', np.zeros((0, 8)))

    out_dir = Path(args.out).parent / f"sample_{args.sample_idx:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_dir = {cls: out_dir / cls for cls in cfg.CLASS_NAMES}
    for p in per_class_dir.values():
        p.mkdir(exist_ok=True, parents=True)

    combined_class_importance = {cls: np.zeros(num_points, dtype=np.float32) for cls in cfg.CLASS_NAMES}
    detection_meta_list = []

    objects_per_class = {cls: 0 for cls in cfg.CLASS_NAMES}
    kept_det_indices = []
    for d_idx in range(len(boxes)):
        det_score = float(scores[d_idx])
        det_label = int(labels[d_idx])
        if det_score < float(args.min_score):
            continue
        class_name = (cfg.CLASS_NAMES[det_label - 1]
                      if 0 <= det_label - 1 < len(cfg.CLASS_NAMES) else f"cls{det_label}")
        objects_per_class[class_name] += 1
        kept_det_indices.append(d_idx)

    print("\n=== LIME Sample Summary ===")
    print(f"sample_idx={args.sample_idx:06d} | num_points={num_points} | detections_kept={len(kept_det_indices)}/{len(boxes)} | min_score={args.min_score}")
    for cls in cfg.CLASS_NAMES:
        print(f"{cls}: {objects_per_class[cls]}")
    print("===========================\n")

    global_ready = False
    num_clusters_global = 0
    cluster_labels_global = None
    cluster_pts_global = None

    def ensure_global_clustering():
        nonlocal global_ready, num_clusters_global, cluster_labels_global, cluster_pts_global
        if global_ready:
            return
        Np = points.shape[0]
        if Np <= 0:
            raise RuntimeError("No points available for global fallback clustering.")

        num_clusters_global = int(min(args.num_clusters, Np))
        num_clusters_global = max(1, num_clusters_global)

        init_centers = fps_init_centers(points[:, :3], num_clusters_global)
        if init_centers.shape[0] < num_clusters_global:
            extra = points[RNG.choice(Np, num_clusters_global - init_centers.shape[0], replace=False), :3]
            init_centers = np.vstack([init_centers, extra])

        kmeans_global = KMeans(
            n_clusters=num_clusters_global,
            init=init_centers,
            n_init=1,
            max_iter=30,
            random_state=args.seed
        ).fit(points[:, :3])

        cluster_labels_global = kmeans_global.labels_
        cluster_pts_global = [points[cluster_labels_global == i] for i in range(num_clusters_global)]
        global_ready = True
        logger.info(f"[GLOBAL] Built fallback clustering: K={num_clusters_global} on N={Np} points")

    for d_idx in range(len(boxes)):
        det_box = boxes[d_idx][:7].astype(np.float32)
        det_score = float(scores[d_idx])
        det_label = int(labels[d_idx])
        class_name = (cfg.CLASS_NAMES[det_label - 1]
                      if 0 <= det_label - 1 < len(cfg.CLASS_NAMES) else f"cls{det_label}")

        if det_score < float(args.min_score):
            continue

        RNG_EVAL = np.random.RandomState(int(args.seed) * 100000 + int(d_idx))

        use_object_centric = bool(args.object_centric)

        inside_mask = None
        inside_pts = None
        outside_pts = None
        inside_cluster_labels = None
        inside_cluster_pts = None

        expand_used = None
        k_used = None
        fallback_reason = None

        if args.object_centric:
            base_expand = choose_expand_base(class_name)
            base_expand = max(base_expand, float(args.box_expand))

            base_mask = points_in_box_mask(points[:, :3], det_box, expand=base_expand)
            base_inside_count = int(base_mask.sum())

            if args.adaptive:
                expand_used = choose_expand_adaptive(
                    class_name=class_name,
                    inside_count_at_base=base_inside_count,
                    base_expand=base_expand,
                    expand_min=args.expand_min,
                    expand_max=args.expand_max
                )
            else:
                expand_used = float(np.clip(base_expand, args.expand_min, args.expand_max))

            inside_mask = points_in_box_mask(points[:, :3], det_box, expand=expand_used)
            inside_pts = points[inside_mask]
            outside_pts = points[~inside_mask]
            inside_count = int(inside_pts.shape[0])

            if inside_count < int(args.min_inside_pts):
                use_object_centric = False
                fallback_reason = f"inside_count={inside_count} < min_inside_pts={args.min_inside_pts}"
                ensure_global_clustering()
            else:
                use_object_centric = True
                k_used = choose_num_clusters(
                    class_name=class_name,
                    inside_count=inside_count,
                    max_clusters=args.num_clusters,
                    target_pts_per_cluster=args.target_ppc,
                    kmin=args.kmin
                )
                k_used = max(1, min(k_used, inside_count))

                init_centers = fps_init_centers(inside_pts[:, :3], k_used)
                if init_centers.shape[0] < k_used:
                    extra = inside_pts[RNG.choice(inside_count, k_used - init_centers.shape[0], replace=False), :3]
                    init_centers = np.vstack([init_centers, extra])

                kmeans_inside = KMeans(
                    n_clusters=k_used,
                    init=init_centers,
                    n_init=1,
                    max_iter=30,
                    random_state=args.seed
                ).fit(inside_pts[:, :3])

                inside_cluster_labels = kmeans_inside.labels_
                inside_cluster_pts = [inside_pts[inside_cluster_labels == i] for i in range(k_used)]

        if not use_object_centric:
            ensure_global_clustering()
            k_used = int(num_clusters_global)

            if inside_mask is None:
                base_expand = choose_expand_base(class_name)
                base_expand = max(base_expand, float(args.box_expand))
                expand_used = float(np.clip(base_expand, args.expand_min, args.expand_max))
                inside_mask = points_in_box_mask(points[:, :3], det_box, expand=expand_used)
                inside_pts = points[inside_mask]

        num_clusters_local = int(k_used)
        num_clusters_local = max(1, num_clusters_local)

        min_keep = max(1, int(args.min_keep_frac * num_clusters_local))
        min_keep = min(min_keep, num_clusters_local)

        masks = []
        for _ in range(int(args.num_perturbs)):
            kkeep = RNG.randint(min_keep, num_clusters_local + 1)
            keep_ids = RNG.choice(num_clusters_local, kkeep, replace=False)
            masks.append(keep_ids)

        pert_scores = np.zeros(int(args.num_perturbs), dtype=np.float32)

        for start in range(0, int(args.num_perturbs), int(args.batch_size)):
            end = min(start + int(args.batch_size), int(args.num_perturbs))

            if use_object_centric:
                batch_pts_list = [
                    safe_build_pert_object_centric(outside_pts, inside_cluster_pts, masks[i], rng=RNG)
                    for i in range(start, end)
                ]
            else:
                batch_pts_list = [
                    safe_build_pert_global(cluster_pts_global, masks[i], rng=RNG)
                    for i in range(start, end)
                ]

            batch_dict = collate_and_pad(batch_pts_list)
            preds = run_model_batch(model, batch_dict)

            for local_i, global_i in enumerate(range(start, end)):
                pred = preds[local_i]

                if pred['pred_boxes'].shape[0] == 0:
                    pert_scores[global_i] = 0.0
                    continue

                pboxes = pred['pred_boxes'].detach().cpu().numpy()
                pscores = pred['pred_scores'].detach().cpu().numpy()
                plabels = pred['pred_labels'].detach().cpu().numpy()

                same_class = plabels == det_label
                if not np.any(same_class):
                    pert_scores[global_i] = 0.0
                    continue

                pboxes_f = pboxes[same_class]
                pscores_f = pscores[same_class]

                match_idx, _ = iou3d_match(det_box, pboxes_f, thresh=float(args.tau_iou))
                pert_scores[global_i] = float(pscores_f[match_idx]) if match_idx != -1 else 0.0

        X = np.zeros((int(args.num_perturbs), num_clusters_local), dtype=np.float32)
        for i, keep in enumerate(masks):
            X[i, keep] = 1.0

        ridge = Ridge(alpha=1.0, random_state=args.seed, fit_intercept=True)
        ridge.fit(X, pert_scores)

        importance_cluster = ridge.coef_.astype(np.float32)
        intercept = float(ridge.intercept_)
        r2 = float(ridge.score(X, pert_scores))

        if use_object_centric:
            point_importance = np.zeros(num_points, dtype=np.float32)
            point_importance[inside_mask] = importance_cluster[inside_cluster_labels]

            cluster_labels_full = np.full(num_points, -1, dtype=np.int32)
            cluster_labels_full[inside_mask] = inside_cluster_labels.astype(np.int32)
            num_clusters_saved = int(num_clusters_local)
        else:
            point_importance = importance_cluster[cluster_labels_global]
            cluster_labels_full = cluster_labels_global.astype(np.int32)
            num_clusters_saved = int(num_clusters_local)

        abs_imp_full = np.abs(point_importance).astype(np.float32)

        focus_ratio = 0.0
        sparsity = 0.0
        delta_s_alpha = 0.0
        del_auc = 0.0
        baseline_eval = 0.0
        fracs_used = []
        del_scores = []

        if args.compute_metrics:
            N0 = int(points.shape[0])
            if N0 > FIXED_NUM_POINTS:
                eval_idx = RNG_EVAL.choice(N0, FIXED_NUM_POINTS, replace=False)
            else:
                eval_idx = np.arange(N0, dtype=np.int64)

            points_eval = points[eval_idx]
            points_xyz_eval = points_eval[:, :3]
            abs_imp_eval = abs_imp_full[eval_idx]

            region_eval = points_in_box_mask(points_xyz_eval, det_box, expand=float(args.focus_expand))
            focus_ratio = compute_focus_ratio(abs_imp_eval, region_eval)

            sparsity = compute_sparsity_all_points(abs_imp_eval, float(args.metrics_alpha))

            pts_base = pad_to_fixed(points_eval, FIXED_NUM_POINTS, rng=RNG_EVAL)
            baseline_eval = float(score_same_detection_strict(
                model=model,
                pts_fixed_xyzi=pts_base,
                det_box7=det_box,
                det_label=det_label,
                iou_thresh=float(args.tau_iou)
            ))

            alpha = float(args.metrics_alpha)
            Ne = int(abs_imp_eval.shape[0])
            k = int(max(1, round(alpha * Ne)))
            top_idx = np.argpartition(abs_imp_eval, -k)[-k:]

            keep_mask = np.ones(Ne, dtype=bool)
            keep_mask[top_idx] = False

            pts_del = points_eval[keep_mask]
            pts_del = pad_to_fixed(pts_del, FIXED_NUM_POINTS, rng=RNG_EVAL)
            s_del = float(score_same_detection_strict(
                model=model,
                pts_fixed_xyzi=pts_del,
                det_box7=det_box,
                det_label=det_label,
                iou_thresh=float(args.tau_iou)
            ))
            delta_s_alpha = float(baseline_eval - s_del)

            del_scores, del_auc, fracs_used = deletion_curve_strict(
                model=model,
                points_eval_xyzi=points_eval,
                abs_importance_eval=abs_imp_eval,
                det_box7=det_box,
                det_label=det_label,
                fracs=del_fracs,
                iou_thresh=float(args.tau_iou),
                rng=RNG_EVAL,
                batch_limit=max(1, int(args.deletion_batch))
            )

        cls_dir = per_class_dir.get(class_name, out_dir / class_name)
        cls_dir.mkdir(parents=True, exist_ok=True)
        det_dir = cls_dir / f"det{d_idx:03d}"
        det_dir.mkdir(parents=True, exist_ok=True)

        npz_path = det_dir / f"lime_sample{args.sample_idx}_det{d_idx:03d}.npz"

        topk = min(10, importance_cluster.shape[0])
        order = np.argsort(np.abs(importance_cluster))[::-1]
        top_clusters = order[:topk].astype(np.int32)
        top_cluster_weights = importance_cluster[top_clusters].astype(np.float32)

        np.savez_compressed(
            npz_path,
            points=points.astype(np.float32),
            point_importance=point_importance.astype(np.float32),
            cluster_labels=cluster_labels_full.astype(np.int32),
            num_clusters=np.int32(num_clusters_saved),
            top_clusters=top_clusters,
            top_cluster_weights=top_cluster_weights,
            pred_box=det_box.astype(np.float32),
            pred_score=np.float32(det_score),
            pred_label=np.int32(det_label),
            intercept=np.float32(intercept),
            r2=np.float32(r2),
            object_centric=np.int32(1 if use_object_centric else 0),
            expand_used=np.float32(expand_used) if expand_used is not None else np.float32(-1.0),
            inside_points=np.int32(int(inside_pts.shape[0]) if inside_pts is not None else 0),

            metrics_alpha=np.float32(args.metrics_alpha),
            focus_expand=np.float32(args.focus_expand),
            tau_iou=np.float32(args.tau_iou),

            focus_ratio=np.float32(focus_ratio),
            sparsity=np.float32(sparsity),
            baseline_eval=np.float32(baseline_eval),
            delta_s_alpha=np.float32(delta_s_alpha),
            deletion_fracs=np.array(fracs_used, dtype=np.float32),
            deletion_scores=np.array(del_scores, dtype=np.float32),
            deletion_auc=np.float32(del_auc),

            fallback_reason=np.array(fallback_reason if fallback_reason else "", dtype=object),
            eval_idx=np.array(eval_idx, dtype=np.int64) if args.compute_metrics else np.array([], dtype=np.int64),
        )

        logger.info(f" → NPZ saved: {npz_path}")

        combined_class_importance[class_name] = np.maximum(
            np.abs(combined_class_importance[class_name]),
            np.abs(point_importance)
        )

        if gt_boxes.shape[0] > 0:
            matched_gt_idx, matched_gt_iou = iou3d_match(det_box, gt_boxes[:, :7])
        else:
            matched_gt_idx, matched_gt_iou = -1, 0.0

        if args.compute_metrics:
            flag = " [WARN baseline_eval=0]" if float(baseline_eval) <= 1e-6 else ""
            print(
                f"[DET] idx={d_idx:03d} cls={class_name:<10} "
                f"pred_score={det_score:.3f} baseline_eval={float(baseline_eval):.3f} "
                f"Focus={float(focus_ratio):.3f} Spars={float(sparsity):.3f} "
                f"dS@{args.metrics_alpha:.2f}={float(delta_s_alpha):.3f} AUC={float(del_auc):.3f}{flag}"
            )

        detection_meta_list.append({
            "det_idx": int(d_idx),
            "class_name": class_name,
            "score": float(det_score),
            "baseline_eval": float(baseline_eval) if args.compute_metrics else None,
            "object_centric": bool(use_object_centric),
            "expand_used": float(expand_used) if expand_used is not None else None,
            "num_clusters": int(num_clusters_local),
            "inside_points": int(inside_pts.shape[0]) if inside_pts is not None else None,
            "target_ppc": int(args.target_ppc),
            "fallback_reason": fallback_reason,
            "r2": float(r2),

            "focus_ratio": float(focus_ratio) if args.compute_metrics else None,
            "sparsity_alpha": float(args.metrics_alpha),
            "sparsity": float(sparsity) if args.compute_metrics else None,
            "delta_s_alpha": float(delta_s_alpha) if args.compute_metrics else None,
            "deletion_auc": float(del_auc) if args.compute_metrics else None,

            "npz": str(npz_path.relative_to(out_dir)),
            "folder": str(det_dir.relative_to(out_dir)),
            "sample_dir": str(out_dir),

            "matched_gt_idx": int(matched_gt_idx),
            "matched_gt_iou": float(matched_gt_iou),
            "eval_size": int(len(eval_idx)) if args.compute_metrics else 0,
        })

    filters = {
        "min_score": float(args.min_score),
        "tau_iou": float(args.tau_iou),
        "fixed_num_points": int(FIXED_NUM_POINTS),
        "metrics_alpha": float(args.metrics_alpha),
        "focus_expand": float(args.focus_expand),
        "deletion_fracs": [float(x) for x in del_fracs],
    }
    save_metadata_json(
        sample_idx=args.sample_idx,
        points=points,
        detections=detection_meta_list,
        class_names=cfg.CLASS_NAMES,
        out_dir=out_dir,
        object_centric=args.object_centric,
        filters=filters
    )

    save_metrics_csv(out_dir, detection_meta_list)

    combined_npz_path = out_dir / f"lime_sample{args.sample_idx}_all_classes.npz"
    all_importances = np.vstack([combined_class_importance[cls] for cls in cfg.CLASS_NAMES]).astype(np.float32)

    np.savez_compressed(
        combined_npz_path,
        points=points.astype(np.float32),
        per_class_importance=all_importances,
        class_names=np.array(cfg.CLASS_NAMES, dtype=object),
        object_centric=np.int32(1 if args.object_centric else 0),
    )
    logger.info(f" → Combined multi-class NPZ: {combined_npz_path}")

    print("\n=== DONE ===")
    print(f"Saved outputs to: {out_dir}")
    print(f"Saved metrics CSV: {out_dir / 'metrics_summary.csv'}")
    print("============\n")


if __name__ == '__main__':
    main()