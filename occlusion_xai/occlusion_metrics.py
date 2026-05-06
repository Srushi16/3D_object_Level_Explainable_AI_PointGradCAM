#!/usr/bin/env python3
# "SOTA-style" point masking explanation (deletion + optional insertion)
# UPDATED (2026-01) + FIXED METRICS (2026-01-21):
# - Deterministic RNG everywhere (grid shuffle, padding/downsample, eval subset)
# - Metrics aligned with LIME/GradCAM evaluation:
#   (1) Focus Ratio (↑ better)
#   (2) Sparsity@alpha on SUPPORT (↑ better, avoids artificial 1.0 from zeros)
#   (3) Faithfulness ΔScore@alpha using BASELINE_EVAL (↑ better)
#   (4) Faithfulness Deletion Curve + AUC (↓ better, remaining score vs removed fraction)
# - Fixes bug: ΔScore@alpha masking now computed on EVAL SET (not full set)
# - Prints per sample: class counts + per-detection metric line

import argparse
from pathlib import Path
import numpy as np
import torch
import logging
import csv
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


# -----------------------------
# -----------------------------
logger = logging.getLogger("SOTA_POINT_MASKING")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# Deterministic RNG (used everywhere)
SEED = 42
RNG = np.random.RandomState(SEED)


def np_to_torch(arr):
    return torch.from_numpy(arr).float().cuda()


def ensure_xyzi(pts: np.ndarray) -> np.ndarray:
    """Ensure (N,4) points: x,y,z,intensity (or any 4th feature)."""
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Expected points shape (N,>=4), got {pts.shape}")
    return pts[:, :4].astype(np.float32, copy=False)


def pad_or_sample_to(pts_xyzi: np.ndarray, n_target: int = 16384) -> np.ndarray:
    """
    Deterministic pad/downsample to n_target points using RNG.
    If n_target <= 0: return input as-is.
    Avoid empty clouds.
    """
    pts_xyzi = ensure_xyzi(pts_xyzi)

    if n_target is None or int(n_target) <= 0:
        return pts_xyzi

    n_target = int(n_target)
    n = int(len(pts_xyzi))

    if n == 0:
        return np.zeros((1, 4), dtype=np.float32)

    if n == n_target:
        return pts_xyzi
    if n > n_target:
        idx = RNG.choice(n, n_target, replace=False)
        return pts_xyzi[idx]

    idx = RNG.choice(n, n_target - n, replace=True)
    return np.concatenate([pts_xyzi, pts_xyzi[idx]], axis=0)


def create_batch(points_list, n_target: int = 16384):
    """
    Fixed-size batching:
      - pad/downsample each cloud to n_target
      - add batch idx column
    Returns:
      points_with_batch: (B*n_target, 5) tensor [batch,x,y,z,i]
      batch_cnt: (B,) int32 = n_target
    """
    padded = []
    for pts in points_list:
        pts = ensure_xyzi(pts)
        pts = pad_or_sample_to(pts, n_target=n_target)
        padded.append(pts)

    batch_array = np.stack(padded)  # (B,n_target,4)
    batch_tensor = torch.from_numpy(batch_array).float().cuda()

    B = len(points_list)
    batch_idx = torch.arange(B, device='cuda').view(-1, 1, 1).repeat(1, n_target, 1)
    points_with_batch = torch.cat([batch_idx, batch_tensor], dim=-1).reshape(-1, 5)
    batch_cnt = torch.full((B,), n_target, dtype=torch.int32, device='cuda')
    return points_with_batch, batch_cnt


# Occlusion matching score drop (used only to build importance map)
def compute_drop(orig_box7, orig_score, orig_label, pred_dict, iou_thr=0.5):
    """
    Compare original detection to masked output:
    - if same class & IoU >= thr found => score drop = orig_score - best_score
    - else => drop = orig_score (object gone)
    Also returns indicator 1.0 if gone else 0.0
    """
    if len(pred_dict['pred_scores']) == 0:
        return float(orig_score), 1.0

    boxes = pred_dict['pred_boxes'].detach().cpu().numpy()
    scores = pred_dict['pred_scores'].detach().cpu().numpy()
    labels = pred_dict['pred_labels'].detach().cpu().numpy()

    ious = iou3d_nms_utils.boxes_iou3d_gpu(
        np_to_torch(boxes),
        np_to_torch(orig_box7[None])
    )[0].detach().cpu().numpy().flatten()

    match = (ious >= float(iou_thr)) & (labels == int(orig_label))
    if match.any():
        new_best_score = float(scores[match].max())
        score_drop = max(0.0, float(orig_score) - new_best_score)
        return float(score_drop), 0.0

    return float(orig_score), 1.0


# Metrics helpers (aligned)
def points_in_box_mask(points_xyz: np.ndarray, box7: np.ndarray, expand: float = 1.0) -> np.ndarray:
    """Oriented 3D box inclusion test."""
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


def compute_focus_ratio(abs_importance: np.ndarray, region_mask: np.ndarray) -> float:
    denom = float(abs_importance.sum())
    if denom <= 1e-12:
        return float("nan")
    return float(abs_importance[region_mask].sum()) / denom


def compute_sparsity(abs_importance: np.ndarray, alpha: float = 0.05) -> float:
    denom = float(abs_importance.sum())
    if denom <= 1e-12:
        return float("nan")
    N = abs_importance.shape[0]
    k = int(max(1, round(alpha * N)))
    idx = np.argpartition(abs_importance, -k)[-k:]
    return float(abs_importance[idx].sum()) / denom


def score_same_detection_strict(model, base_dict, pts_xyzi: np.ndarray, det_box7: np.ndarray, det_label: int,
                                iou_thr: float = 0.5) -> float:
    """
    STRICT match score:
      same class AND IoU >= iou_thr => return score (best IoU)
      else => 0.0
    """
    pts_xyzi = ensure_xyzi(pts_xyzi)
    if pts_xyzi.shape[0] == 0:
        pts_xyzi = np.zeros((1, 4), dtype=np.float32)

    pts_xyzi = pts_xyzi.astype(np.float32, copy=False)
    batch_idx_col = np.zeros((pts_xyzi.shape[0], 1), dtype=np.float32)
    pts_with_batch = np.hstack([batch_idx_col, pts_xyzi])  # (N,5)

    batch = base_dict.copy()
    batch['points'] = torch.from_numpy(pts_with_batch).float().cuda()
    batch['batch_size'] = 1
    batch['xyz_batch_cnt'] = torch.tensor([len(pts_xyzi)], device='cuda', dtype=torch.int32)

    load_data_to_gpu(batch)

    with torch.no_grad():
        pred_dicts, _ = model(batch)

    pred = pred_dicts[0]
    if len(pred['pred_scores']) == 0:
        return 0.0

    boxes = pred['pred_boxes'].detach().cpu().numpy()
    scores = pred['pred_scores'].detach().cpu().numpy()
    labels = pred['pred_labels'].detach().cpu().numpy()

    same = (labels == int(det_label))
    if not np.any(same):
        return 0.0

    boxes_f = boxes[same]
    scores_f = scores[same]

    ious = iou3d_nms_utils.boxes_iou3d_gpu(
        torch.from_numpy(boxes_f).float().cuda(),
        torch.from_numpy(det_box7[None]).float().cuda()
    )[0].detach().cpu().numpy().flatten()

    best = int(np.argmax(ious))
    if float(ious[best]) >= float(iou_thr):
        return float(scores_f[best])
    return 0.0


def deletion_curve_strict(model, base_dict, points_xyzi: np.ndarray, abs_importance: np.ndarray,
                          det_box7: np.ndarray, det_label: int,
                          fracs, iou_thr: float = 0.5, n_target: int = 16384):
    """
    Remove top frac points by |importance| on the SAME evaluation set.
    STRICT matching returns 0 if not matched.
    Returns (fracs_list, scores_list, auc)
    """
    points_xyzi = ensure_xyzi(points_xyzi)
    abs_importance = np.asarray(abs_importance, dtype=np.float32)

    N = abs_importance.shape[0]
    if N != int(points_xyzi.shape[0]):
        raise ValueError(f"abs_importance length {N} != points length {points_xyzi.shape[0]}")

    fracs_list = [float(x) for x in fracs]
    if len(fracs_list) == 0 or fracs_list[0] != 0.0:
        fracs_list = [0.0] + fracs_list

    order = np.argsort(-abs_importance)  # descending by importance
    scores_list = []

    for f in fracs_list:
        k = int(round(float(f) * N))
        if k <= 0:
            pts_keep = points_xyzi
        else:
            remove_idx = order[:k]
            keep_mask = np.ones(N, dtype=bool)
            keep_mask[remove_idx] = False
            pts_keep = points_xyzi[keep_mask]

        pts_keep = pad_or_sample_to(pts_keep, n_target=int(n_target))
        if pts_keep.shape[0] == 0:
            pts_keep = np.zeros((1, 4), dtype=np.float32)

        s = score_same_detection_strict(model, base_dict, pts_keep, det_box7, det_label, iou_thr=float(iou_thr))
        scores_list.append(float(s))

    x = np.array(fracs_list, dtype=np.float32)
    y = np.array(scores_list, dtype=np.float32)
    auc = float(np.trapz(y, x))  # lower is better
    return fracs_list, scores_list, auc


def main():
    parser = argparse.ArgumentParser(description="SOTA Point Masking Explainer + LIME-aligned metrics (fixed)")
    parser.add_argument('--cfg', required=True)
    # UPDATE PATH: --cfg should point to your OpenPCDet config YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    parser.add_argument('--ckpt', required=True)
    # UPDATE PATH: --ckpt should point to your trained model checkpoint (.pth)
    parser.add_argument('--root_path', type=str, required=True)
    # UPDATE PATH: --root_path should point to the KITTI dataset root (as used by OpenPCDet)
    parser.add_argument('--info_dir', type=str, required=True)
    # UPDATE PATH: --info_dir should point to the directory containing KITTI info files (e.g., kitti_infos_test.pkl)
    parser.add_argument('--sample_idx', type=int, required=True)

    parser.add_argument('--num_masks', type=int, default=8000)
    parser.add_argument('--mask_scales', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    parser.add_argument('--use_insertion', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out', type=str, default='sota_explanations')
    # UPDATE PATH: --out is the output folder where explanations/metrics will be written

    # Metrics controls
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--metrics_alpha', type=float, default=0.05)
    parser.add_argument('--deletion_fracs', type=float, nargs='+',
                        default=[0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument('--focus_expand', type=float, default=1.6)
    parser.add_argument('--n_target', type=int, default=16384)

    args = parser.parse_args()

    # Load config, dataset and model
    cfg_from_yaml_file(args.cfg, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.root_path
    cfg.DATA_CONFIG.INFO_DIR = args.info_dir

    from pcdet.datasets.kitti.kitti_zip_dataset import KittiZipDataset
    dataset = KittiZipDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.root_path),
        logger=common_utils.create_logger()
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=common_utils.create_logger())
    model.cuda().eval()

    # Load frame
    data_dict = dataset[args.sample_idx]
    if data_dict is None:
        raise ValueError(f"Sample {args.sample_idx} is empty")

    points_np_full = data_dict['points'].copy()     # (N,>=4)
    points_xyzi = ensure_xyzi(points_np_full)       # (N,4)
    points_xyz = points_xyzi[:, :3]                 # (N,3)

    base_dict = {k: v for k, v in data_dict.items() if k not in ('gt_boxes', 'gt_names', 'gt_boxes2d')}

    # One forward pass for original detections (fixed-size input)
    pts_one = pad_or_sample_to(points_xyzi, n_target=int(args.n_target))
    batch_idx_col = np.zeros((pts_one.shape[0], 1), dtype=np.float32)
    points_with_batch_np = np.hstack([batch_idx_col, pts_one])  # (n_target,5)

    orig_batch = base_dict.copy()
    orig_batch['points'] = torch.from_numpy(points_with_batch_np).float().cuda()
    orig_batch['batch_size'] = 1
    orig_batch['xyz_batch_cnt'] = torch.tensor([len(pts_one)], device='cuda', dtype=torch.int32)
    load_data_to_gpu(orig_batch)

    with torch.no_grad():
        pred_dicts, _ = model(orig_batch)
    orig_pred = pred_dicts[0]

    boxes = orig_pred['pred_boxes'].detach().cpu().numpy()
    scores = orig_pred['pred_scores'].detach().cpu().numpy()
    labels = orig_pred['pred_labels'].detach().cpu().numpy()

    # Print per-sample detection summary (after score_thr)
    objects_per_class = {cls: 0 for cls in cfg.CLASS_NAMES}
    kept = []
    for det_idx in range(len(boxes)):
        s = float(scores[det_idx])
        if s < float(args.score_thr):
            continue
        lab = int(labels[det_idx])
        cls_name = cfg.CLASS_NAMES[lab - 1] if 1 <= lab <= len(cfg.CLASS_NAMES) else f"cls{lab}"
        objects_per_class[cls_name] += 1
        kept.append(det_idx)

    print("\n=== Occlusion Sample Summary ===")
    print(f"sample_idx={args.sample_idx:06d} | num_points={len(points_xyzi)} | kept={len(kept)}/{len(boxes)} | score_thr={args.score_thr}")
    for cls in cfg.CLASS_NAMES:
        print(f"{cls}: {objects_per_class[cls]}")
    print("================================\n")

    # Output folder for this sample
    out_root = Path(args.out) / f"sample_{args.sample_idx:06d}"
    out_root.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_root / "metrics_summary.csv"
    write_header = (not metrics_csv.exists())

    rows = []

    # Explain detections
    for det_idx in range(len(boxes)):
        score = float(scores[det_idx])
        label = int(labels[det_idx])
        box = boxes[det_idx].astype(np.float32)

        if score < float(args.score_thr):
            continue

        cls_name = cfg.CLASS_NAMES[label - 1]
        logger.info(f"Explaining {cls_name} det{det_idx:03d} — pred_score={score:.3f}")

        # Per-point importance on ORIGINAL points
        importance = np.zeros(len(points_xyz), dtype=np.float32)

        center = box[:3]
        size = box[3:6]

        # 1) Build grid centers around object (deterministic shuffle)
        expand_grid = 1.6
        xs = np.arange(center[0] - size[0] * expand_grid, center[0] + size[0] * expand_grid, 0.4)
        ys = np.arange(center[1] - size[1] * expand_grid, center[1] + size[1] * expand_grid, 0.4)
        zs = np.arange(center[2] - size[2] * expand_grid, center[2] + size[2] * 1.5, 0.4)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        perm = RNG.permutation(grid_centers.shape[0])
        grid_centers = grid_centers[perm]

        centers_per_scale = grid_centers[: max(1, int(args.num_masks) // max(1, len(args.mask_scales)))]

        # 2) Generate masked clouds
        batch_pts_list = []
        mask_info = []  # ('del'/'ins', boolean mask on ORIGINAL points)

        for scale in args.mask_scales:
            half = float(scale) / 2.0
            for c in centers_per_scale:
                lo = c - half
                hi = c + half
                inside = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)

                masked_del = points_xyzi[~inside]
                if masked_del.shape[0] > 10:
                    batch_pts_list.append(masked_del)
                    mask_info.append(('del', inside))

                if args.use_insertion:
                    masked_ins = points_xyzi[inside]
                    if masked_ins.shape[0] > 10:
                        batch_pts_list.append(masked_ins)
                        mask_info.append(('ins', inside))

        # 3) Run model on all masks in batches
        drops = []
        for i in tqdm(range(0, len(batch_pts_list), int(args.batch_size)),
                      desc=f"{cls_name} det{det_idx}", leave=False):
            batch_pts = batch_pts_list[i:i + int(args.batch_size)]
            points_with_batch, batch_cnt = create_batch(batch_pts, n_target=int(args.n_target))

            cur_batch = base_dict.copy()
            cur_batch['points'] = points_with_batch
            cur_batch['batch_size'] = len(batch_pts)
            cur_batch['xyz_batch_cnt'] = batch_cnt
            load_data_to_gpu(cur_batch)

            with torch.no_grad():
                pred_dicts, _ = model(cur_batch)

            for pred in pred_dicts:
                s_drop, i_drop = compute_drop(box[:7], score, label, pred, iou_thr=float(args.iou_thr))
                drops.append(0.7 * float(s_drop) + 0.3 * float(i_drop))

        # 4) Accumulate importance on ORIGINAL points
        for drop, (mode, mask) in zip(drops, mask_info):
            if mode == 'del':
                importance[mask] += float(drop)
            else:
                importance[mask] += (1.0 - float(drop))

        # Normalize for viz comparability (metrics use abs anyway)
        if float(importance.max()) > 0:
            importance /= float(importance.max())

        abs_imp_full = np.abs(importance).astype(np.float32)

        # Fixed evaluation subset ONCE (deterministic) for all metrics
        N0 = int(points_xyzi.shape[0])
        if int(args.n_target) > 0 and N0 > int(args.n_target):
            eval_idx = RNG.choice(N0, int(args.n_target), replace=False)
        else:
            eval_idx = np.arange(N0, dtype=np.int64)

        points_xyzi_eval = points_xyzi[eval_idx]
        points_xyz_eval = points_xyz[eval_idx]
        abs_imp_eval = abs_imp_full[eval_idx]

        # METRICS (fixed & aligned)
        # Focus computed on eval set (consistent)
        region_eval = points_in_box_mask(points_xyz_eval, box[:7], expand=float(args.focus_expand))
        focus_ratio = compute_focus_ratio(abs_imp_eval, region_eval)

        # Sparsity on SUPPORT (avoid artificial 1.0 from zeros)
        nz = abs_imp_eval > 0
        abs_for_spars = abs_imp_eval[nz] if np.any(nz) else abs_imp_eval
        sparsity = compute_sparsity(abs_for_spars, alpha=float(args.metrics_alpha))

        # Baseline eval score computed on same evaluation input distribution
        pts_base = pad_or_sample_to(points_xyzi_eval, n_target=int(args.n_target))
        baseline_eval = score_same_detection_strict(
            model=model, base_dict=base_dict, pts_xyzi=pts_base,
            det_box7=box[:7], det_label=label, iou_thr=float(args.iou_thr)
        )

        # ΔScore@alpha (STRICT) computed on eval set (FIXED BUG)
        N = int(abs_imp_eval.shape[0])
        k = int(max(1, round(float(args.metrics_alpha) * N)))
        top_idx = np.argpartition(abs_imp_eval, -k)[-k:]

        keep_mask = np.ones(N, dtype=bool)
        keep_mask[top_idx] = False

        pts_del = points_xyzi_eval[keep_mask]
        pts_del = pad_or_sample_to(pts_del, n_target=int(args.n_target))

        s_del = score_same_detection_strict(
            model=model, base_dict=base_dict, pts_xyzi=pts_del,
            det_box7=box[:7], det_label=label, iou_thr=float(args.iou_thr)
        )
        delta_s_alpha = float(baseline_eval - s_del)

        # Deletion curve + AUC on eval set
        del_fracs, del_scores, del_auc = deletion_curve_strict(
            model=model, base_dict=base_dict,
            points_xyzi=points_xyzi_eval, abs_importance=abs_imp_eval,
            det_box7=box[:7], det_label=label,
            fracs=[float(x) for x in args.deletion_fracs],
            iou_thr=float(args.iou_thr), n_target=int(args.n_target)
        )

        # Print per detection line (what you asked)
        warn = " [WARN baseline_eval=0]" if float(baseline_eval) <= 1e-6 else ""
        print(
            f"[DET] idx={det_idx:03d} cls={cls_name:<10} pred_score={score:.3f} "
            f"baseline_eval={float(baseline_eval):.3f} Focus={float(focus_ratio):.3f} "
            f"Spars={float(sparsity):.3f} dS@{args.metrics_alpha:.2f}={float(delta_s_alpha):.3f} "
            f"AUC={float(del_auc):.3f}{warn}"
        )

        # Save NPZ + CSV row
        save_dir = out_root / cls_name / f"det{det_idx:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_dir / "explanation.npz"
        np.savez_compressed(
            npz_path,
            points=points_xyz.astype(np.float32),
            importance=importance.astype(np.float32),
            box=box.astype(np.float32),
            score=np.float32(score),
            label=np.int32(label),

            # eval subset info
            eval_idx=eval_idx.astype(np.int64),
            n_target=np.int32(args.n_target),

            # metrics
            metrics_alpha=np.float32(args.metrics_alpha),
            focus_expand=np.float32(args.focus_expand),
            iou_thr=np.float32(args.iou_thr),

            focus_ratio=np.float32(focus_ratio),
            sparsity=np.float32(sparsity),
            baseline_eval=np.float32(baseline_eval),
            delta_s_alpha=np.float32(delta_s_alpha),
            deletion_fracs=np.array(del_fracs, dtype=np.float32),
            deletion_scores=np.array(del_scores, dtype=np.float32),
            deletion_auc=np.float32(del_auc),

            sparsity_mode=np.array("support", dtype=object),
        )

        logger.info(f"   Saved → {npz_path}")

        rows.append({
            "sample_idx": int(args.sample_idx),
            "class_name": cls_name,
            "det_idx": int(det_idx),
            "pred_score": float(score),
            "baseline_eval": float(baseline_eval),
            "metrics_alpha": float(args.metrics_alpha),
            "focus_expand": float(args.focus_expand),
            "iou_thr": float(args.iou_thr),
            "focus_ratio": float(focus_ratio) if np.isfinite(focus_ratio) else np.nan,
            "sparsity": float(sparsity) if np.isfinite(sparsity) else np.nan,
            "delta_s_alpha": float(delta_s_alpha),
            "deletion_auc": float(del_auc),
            "npz_path": str(npz_path),
        })

    # Write CSV
    fieldnames = [
        "sample_idx", "class_name", "det_idx",
        "pred_score", "baseline_eval",
        "metrics_alpha", "focus_expand", "iou_thr",
        "focus_ratio", "sparsity", "delta_s_alpha", "deletion_auc",
        "npz_path"
    ]

    if write_header:
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    if rows:
        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for r in rows:
                w.writerow(r)

    logger.info(f"All detections explained — results in {out_root}")
    logger.info(f"Metrics CSV → {metrics_csv}")


if __name__ == '__main__':
    main()
