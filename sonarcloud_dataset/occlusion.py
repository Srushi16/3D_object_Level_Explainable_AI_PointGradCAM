
#!/usr/bin/env python3
"""
Type-A Occlusion Explanation for SonarCloud (OpenPCDet / PointRCNN)

Type-A = score-sensitivity / objectness:
"Which sonar points make the network think there is an object here at all?"

Key properties:
- No IoU matching / no requirement to keep the same predicted box
- Explains a scalar target per masked scene: best detection score
  (optionally constrained to a region near a known object center)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
import time
import traceback

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


logger = logging.getLogger("SONAR_TYPEA_OCCLUSION")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def debug_array(name: str, a: np.ndarray, max_rows: int = 3):
    try:
        logger.info(
            f"[DBG] {name}: shape={a.shape} dtype={a.dtype} "
            f"min={np.nanmin(a):.4f} max={np.nanmax(a):.4f} "
            f"first_rows=\n{a[:max_rows]}"
        )
    except Exception as e:
        logger.info(f"[DBG] {name}: (failed to inspect array) {e}")


def debug_tensor(name: str, t: torch.Tensor, max_elems: int = 5):
    try:
        logger.info(
            f"[DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={float(t.min().item()):.4f} max={float(t.max().item()):.4f} "
            f"sample={t.flatten()[:max_elems].detach().cpu().numpy()}"
        )
    except Exception as e:
        logger.info(f"[DBG] {name}: (failed to inspect tensor) {e}")


def create_batch(points_list, force_fixed_n=False, fixed_n=10000, seed=42):
    """
    Fixed-N batching (optional):
    - Truncate if > fixed_n
    - Duplicate if < fixed_n
    - xyz_batch_cnt must reflect final point count
    """
    rng = np.random.default_rng(seed)
    proc = []
    cnts = []
    for pts in points_list:
        pts = pts[:, :4].astype(np.float32)
        N_orig = pts.shape[0]

        if force_fixed_n:
            if N_orig > fixed_n:
                idx = rng.choice(N_orig, fixed_n, replace=False)
                pts = pts[idx]
                N_final = fixed_n
            elif N_orig < fixed_n:
                n_dup = fixed_n - N_orig
                idx_dup = rng.choice(N_orig, n_dup, replace=True)
                pts_dup = pts[idx_dup]
                pts = np.concatenate([pts, pts_dup], axis=0)
                N_final = fixed_n
            else:
                N_final = N_orig
        else:
            N_final = N_orig

        proc.append(pts)
        cnts.append(N_final)

    batch_cnt = torch.tensor(cnts, dtype=torch.int32, device="cuda")

    out_list = []
    for b, pts in enumerate(proc):
        batch_idx = np.full((pts.shape[0], 1), b, dtype=np.float32)
        out_list.append(np.hstack([batch_idx, pts]).astype(np.float32))

    points_with_batch_np = np.concatenate(out_list, axis=0)
    points_with_batch = torch.from_numpy(points_with_batch_np).float().cuda()

    return points_with_batch, batch_cnt


def load_sonar_points(
    npy_path: str,
    downsample_cap: int = 10000,
    intensity_value: float = 0.5,
    xyz_scale: float = 1.0,
    xyz_shift: list = [0.0, 0.0, 0.0],
    intensity_mode: str = "const",
    add_background: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns (N,4) float32 points: [x,y,z,intensity]
    Downsamples only if N > cap; never upsamples.
    """
    p = Path(npy_path)
    if not p.exists():
        raise FileNotFoundError(f"Sonar npy not found: {npy_path}")

    pts = np.load(p).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected sonar npy shape (N,3+) but got {pts.shape}")

    xyz = pts[:, :3].astype(np.float32)

    xyz = xyz * float(xyz_scale)
    xyz = xyz + np.array(xyz_shift, dtype=np.float32)

    if pts.shape[1] >= 4 and intensity_mode not in ("random", "uniform_high"):
        inten = pts[:, 3:4].astype(np.float32)
    else:
        if intensity_mode == "random":
            rng = np.random.default_rng(seed)
            inten = rng.uniform(0.1, 0.9, (xyz.shape[0], 1)).astype(np.float32)
        elif intensity_mode == "uniform_high":
            inten = np.full((xyz.shape[0], 1), 0.7, dtype=np.float32)
        else:
            inten = np.full((xyz.shape[0], 1), float(intensity_value), dtype=np.float32)

    points_np = np.hstack([xyz, inten]).astype(np.float32)

    finite = np.isfinite(points_np).all(axis=1)
    points_np = points_np[finite]
    nonzero = np.linalg.norm(points_np[:, :3], axis=1) > 1e-6
    points_np = points_np[nonzero]

    if add_background:
        rng = np.random.default_rng(seed)
        obj_center = points_np[:, :3].mean(axis=0)

        plane_half = 12.0
        plane_res = 0.8
        px, py = np.mgrid[
            obj_center[0] - plane_half:obj_center[0] + plane_half:plane_res,
            obj_center[1] - plane_half:obj_center[1] + plane_half:plane_res
        ]
        pz = rng.normal(-0.1, 0.15, px.shape).astype(np.float32)
        pz = np.minimum(pz, obj_center[2] - 0.5).astype(np.float32)

        pi = rng.uniform(0.01, 0.08, px.size).astype(np.float32)
        bg = np.c_[px.ravel(), py.ravel(), pz.ravel(), pi].astype(np.float32)

        n_speckle = int(points_np.shape[0] * 0.06)
        if n_speckle > 0:
            speck_xyz = rng.uniform(
                [obj_center[0] - plane_half, obj_center[1] - plane_half, obj_center[2] - 2.0],
                [obj_center[0] + plane_half, obj_center[1] + plane_half, obj_center[2] + 2.0],
                size=(n_speckle, 3)
            ).astype(np.float32)
            speck_i = rng.uniform(0.0, 0.3, n_speckle).astype(np.float32)
            speck = np.c_[speck_xyz, speck_i].astype(np.float32)
            bg = np.vstack([bg, speck])

        points_np = np.vstack([points_np, bg]).astype(np.float32)
        logger.info(f"[BACKGROUND] Added {len(bg)} background/noise points")

    if downsample_cap is not None and points_np.shape[0] > int(downsample_cap):
        rng = np.random.default_rng(seed)
        idx = rng.choice(points_np.shape[0], size=int(downsample_cap), replace=False)
        points_np = points_np[idx]

    return points_np


def get_target_scalar(pred_dict, target_center=None, target_lwh=None,
                      region_mode="box", region_expand=1.3, region_radius=6.0,
                      class_filter=None):
    """
    Returns:
      scalar_score: best score among candidate detections
      hit_count: number of detections considered (region + class constraints)
      best_idx: index of best box in considered set
    """
    scores_t = pred_dict.get("pred_scores", None)
    boxes_t = pred_dict.get("pred_boxes", None)
    labels_t = pred_dict.get("pred_labels", None)

    if scores_t is None or boxes_t is None or labels_t is None:
        return 0.0, 0, -1

    scores = scores_t.detach().cpu().numpy()
    boxes = boxes_t.detach().cpu().numpy()
    labels = labels_t.detach().cpu().numpy()

    if boxes.shape[0] == 0:
        return 0.0, 0, -1

    keep = np.ones((boxes.shape[0],), dtype=bool)
    if class_filter is not None:
        keep &= (labels == int(class_filter))

    if target_center is not None:
        c = boxes[:, :3]
        if region_mode == "radius":
            d = np.linalg.norm(c - target_center[None, :], axis=1)
            keep &= (d <= float(region_radius))
        else:
            lwh = np.array(target_lwh, dtype=np.float32)
            half = 0.5 * lwh * float(region_expand)
            lo = target_center - half
            hi = target_center + half
            keep &= np.all((c >= lo[None, :]) & (c <= hi[None, :]), axis=1)

    idxs = np.where(keep)[0]
    if idxs.size == 0:
        return 0.0, 0, -1

    best_local = idxs[np.argmax(scores[idxs])]
    return float(scores[best_local]), int(idxs.size), int(best_local)


def save_raw_detection_summary(out_root: Path, pred_dict, class_names):
    scores = pred_dict["pred_scores"].detach().cpu().numpy() if pred_dict.get("pred_scores") is not None else np.array([])
    boxes = pred_dict["pred_boxes"].detach().cpu().numpy() if pred_dict.get("pred_boxes") is not None else np.zeros((0, 7), dtype=np.float32)
    labels = pred_dict["pred_labels"].detach().cpu().numpy() if pred_dict.get("pred_labels") is not None else np.array([], dtype=np.int32)

    order = np.argsort(-scores) if scores.size else np.array([], dtype=np.int64)
    topk = order[:10]

    rows = []
    for i in topk:
        lab = int(labels[i])
        cls = class_names[lab - 1] if 1 <= lab <= len(class_names) else f"cls{lab}"
        b = boxes[i]
        rows.append({
            "rank": int(np.where(topk == i)[0][0]) + 1,
            "idx": int(i),
            "score": float(scores[i]),
            "label_id": lab,
            "class_name": cls,
            "cx": float(b[0]), "cy": float(b[1]), "cz": float(b[2]),
            "l": float(b[3]), "w": float(b[4]), "h": float(b[5]),
            "yaw": float(b[6]) if b.shape[0] > 6 else 0.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "raw_detection_top10.csv", index=False)
    logger.info(f"[INFO] Saved raw_detection_top10.csv ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Type-A Occlusion Explainer for SonarCloud (PointRCNN)")

    parser.add_argument("--cfg", required=True)   # TODO(PATH): model YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    parser.add_argument("--ckpt", required=True)  # TODO(PATH): checkpoint path

    parser.add_argument("--root_path", type=str, required=True)  # TODO(PATH): KITTI root (metadata required by OpenPCDet)
    parser.add_argument("--info_dir", type=str, required=True)   # TODO(PATH): KITTI infos folder (kitti_infos_*.pkl)

    parser.add_argument("--sonar_npy", type=str, required=True)  # TODO(PATH): sonar .npy file (Nx3 or Nx4)
    parser.add_argument("--cap_points", type=int, default=10000)
    parser.add_argument("--score_thresh", type=float, default=1e-6)

    parser.add_argument("--xyz_scale", type=float, default=5.0)
    parser.add_argument("--xyz_shift", type=float, nargs=3, default=[25.0, 0.0, -1.5])
    parser.add_argument("--intensity_mode", choices=["const", "random", "uniform_high"], default="uniform_high")
    parser.add_argument("--sonar_intensity", type=float, default=0.5)
    parser.add_argument("--add_background", action="store_true")

    parser.add_argument("--recon_summary_csv", type=str, default=None)  # TODO(PATH): optional but recommended
    parser.add_argument("--target_region", choices=["global", "box", "radius"], default="box")
    parser.add_argument("--region_expand", type=float, default=1.3)
    parser.add_argument("--region_radius", type=float, default=6.0)
    parser.add_argument("--target_class", type=str, default="any", help="any|Car|Pedestrian|Cyclist")

    parser.add_argument("--num_masks", type=int, default=300)
    parser.add_argument("--mask_scales", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--use_insertion", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--force_fixed_n", action="store_true")
    parser.add_argument("--fixed_n", type=int, default=10000)

    parser.add_argument("--out", type=str, default="sota_typeA_explanations")  # TODO(PATH): output folder
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    set_seeds(args.seed)

    logger.info("========== START (Type A) ==========")
    logger.info(f"Args: {vars(args)}")

    cfg_from_yaml_file(args.cfg, cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = float(args.score_thresh)
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
    logger.info("[INFO] Model loaded and set to eval().")

    sample_tag = Path(args.sonar_npy).stem
    points_np = load_sonar_points(
        npy_path=args.sonar_npy,
        downsample_cap=args.cap_points,
        intensity_value=args.sonar_intensity,
        xyz_scale=args.xyz_scale,
        xyz_shift=args.xyz_shift,
        intensity_mode=args.intensity_mode,
        add_background=args.add_background,
        seed=args.seed
    )
    points_xyz = points_np[:, :3].copy()

    if args.debug:
        debug_array("points_np", points_np, max_rows=5)

    snapped_center = None
    snapped_lwh = None
    if args.recon_summary_csv is not None and Path(args.recon_summary_csv).exists():
        df = pd.read_csv(args.recon_summary_csv)
        row = df[df["name"] == sample_tag]
        if len(row) > 0:
            obj_min = row[["min_x", "min_y", "min_z"]].values[0].astype(np.float32)
            obj_max = row[["max_x", "max_y", "max_z"]].values[0].astype(np.float32)

            obj_min = obj_min * float(args.xyz_scale) + np.array(args.xyz_shift, dtype=np.float32)
            obj_max = obj_max * float(args.xyz_scale) + np.array(args.xyz_shift, dtype=np.float32)

            snapped_center = (obj_min + obj_max) / 2.0
            snapped_lwh = (obj_max - obj_min)
            logger.info(f"[SNAP] Loaded bounds from CSV for {sample_tag}")
            logger.info(f"[SNAP] snapped_center={snapped_center.tolist()}")
            logger.info(f"[SNAP] snapped_lwh={snapped_lwh.tolist()}")
        else:
            logger.info("[SNAP] recon_summary_csv provided but no matching row found; running without snap region.")
    else:
        logger.info("[SNAP] No recon_summary_csv provided/found; running without snap region (global target).")

    base_dict = {"frame_id": sample_tag}

    batch_idx_col = np.zeros((points_np.shape[0], 1), dtype=np.float32)
    points_with_batch_np = np.hstack([batch_idx_col, points_np]).astype(np.float32)

    orig_batch = base_dict.copy()
    orig_batch["points"] = torch.from_numpy(points_with_batch_np).float().cuda()
    orig_batch["batch_size"] = 1
    orig_batch["xyz_batch_cnt"] = torch.tensor([points_np.shape[0]], dtype=torch.int32, device="cuda")
    load_data_to_gpu(orig_batch)

    with torch.no_grad():
        pred_dicts, _ = model(orig_batch)
    orig_pred = pred_dicts[0]

    out_root = Path(args.out) / sample_tag
    out_root.mkdir(parents=True, exist_ok=True)

    boxes = orig_pred["pred_boxes"].detach().cpu().numpy()
    scores = orig_pred["pred_scores"].detach().cpu().numpy()
    labels = orig_pred["pred_labels"].detach().cpu().numpy()

    np.savez_compressed(
        out_root / "orig_pred.npz",
        points=points_xyz.astype(np.float32),
        points_full=points_np.astype(np.float32),
        pred_boxes=boxes.astype(np.float32),
        pred_scores=scores.astype(np.float32),
        pred_labels=labels.astype(np.int32),
        snapped_center=np.array(snapped_center, dtype=np.float32) if snapped_center is not None else np.zeros((3,), dtype=np.float32),
        snapped_lwh=np.array(snapped_lwh, dtype=np.float32) if snapped_lwh is not None else np.zeros((3,), dtype=np.float32),
    )

    logger.info(f"[INFO] Original predictions: {len(boxes)} boxes")
    if len(scores) > 0:
        logger.info(f"[INFO] score stats: min={scores.min():.6f} max={scores.max():.6f}")

    save_raw_detection_summary(out_root, orig_pred, cfg.CLASS_NAMES)

    class_filter = None
    if args.target_class.lower() != "any":
        name_to_id = {n.lower(): (i + 1) for i, n in enumerate(cfg.CLASS_NAMES)}
        key = args.target_class.lower()
        if key not in name_to_id:
            raise ValueError(f"--target_class {args.target_class} not in cfg.CLASS_NAMES={cfg.CLASS_NAMES}")
        class_filter = name_to_id[key]
        logger.info(f"[TARGET] Filtering objectness to class={args.target_class} (label_id={class_filter})")
    else:
        logger.info("[TARGET] objectness uses ANY class (best score).")

    if args.target_region == "global" or snapped_center is None:
        target_center = None
        target_lwh = None
        region_mode = None
        logger.info("[TARGET] Using GLOBAL objectness target (no region filter).")
    else:
        target_center = snapped_center.astype(np.float32)
        target_lwh = snapped_lwh.astype(np.float32)
        region_mode = "radius" if args.target_region == "radius" else "box"
        logger.info(f"[TARGET] Using REGION objectness target: mode={region_mode}")

    orig_scalar, orig_hits, orig_best_idx = get_target_scalar(
        orig_pred,
        target_center=target_center,
        target_lwh=target_lwh,
        region_mode=region_mode if region_mode is not None else "box",
        region_expand=args.region_expand,
        region_radius=args.region_radius,
        class_filter=class_filter
    )
    logger.info(f"[TARGET] orig_scalar={orig_scalar:.6f} (region_hits={orig_hits}, best_idx={orig_best_idx})")

    if orig_scalar <= 1e-9:
        logger.warning(
            "[WARN] orig_scalar ~ 0. The model is not confident under this target. "
            "Explanations will look weak/flat. Consider changing transforms or using target_region=global."
        )

    if target_center is None:
        pc_min = points_xyz.min(axis=0)
        pc_max = points_xyz.max(axis=0)
        center_for_grid = 0.5 * (pc_min + pc_max)
        size_for_grid = (pc_max - pc_min)
        logger.info("[MASK] Using pointcloud bounds for grid (no snap).")
    else:
        center_for_grid = target_center
        size_for_grid = target_lwh
        logger.info("[MASK] Using snapped target bounds for grid.")

    importance = np.zeros((points_xyz.shape[0],), dtype=np.float32)

    expand = 1.6
    step_grid = 0.4
    xs = np.arange(center_for_grid[0] - size_for_grid[0] * expand,
                   center_for_grid[0] + size_for_grid[0] * expand + 1e-6, step_grid)
    ys = np.arange(center_for_grid[1] - size_for_grid[1] * expand,
                   center_for_grid[1] + size_for_grid[1] * expand + 1e-6, step_grid)
    zs = np.arange(center_for_grid[2] - size_for_grid[2] * expand,
                   center_for_grid[2] + size_for_grid[2] * 1.5 + 1e-6, step_grid)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    np.random.shuffle(grid_centers)

    centers_per_scale = grid_centers[:max(1, args.num_masks // max(1, len(args.mask_scales)))]
    logger.info(f"[MASK] grid_centers total={len(grid_centers)} using_per_scale={len(centers_per_scale)} scales={args.mask_scales}")

    batch_pts_list = []
    mask_info = []

    t0 = time.time()
    for scale in args.mask_scales:
        half = float(scale) / 2.0
        for c in centers_per_scale:
            lo = c - half
            hi = c + half
            inside = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)

            masked_del = points_np[~inside]
            if masked_del.shape[0] > 10:
                batch_pts_list.append(masked_del)
                mask_info.append(("del", inside))

            if args.use_insertion:
                masked_ins = points_np[inside]
                if masked_ins.shape[0] > 10:
                    batch_pts_list.append(masked_ins)
                    mask_info.append(("ins", inside))

    logger.info(f"[MASK] Built {len(batch_pts_list)} masks in {time.time()-t0:.2f}s (use_insertion={args.use_insertion})")
    if len(batch_pts_list) == 0:
        raise RuntimeError("No valid masks generated. Try larger mask_scales or smaller step_grid.")

    drops = []
    hit_counts = []

    n_batches = int(np.ceil(len(batch_pts_list) / args.batch_size))
    logger.info(f"[RUN] Masked inference: total_masks={len(batch_pts_list)} batch_size={args.batch_size} -> {n_batches} forward passes")

    for i in tqdm(range(0, len(batch_pts_list), args.batch_size), desc="TypeA masks", leave=False):
        batch_pts = batch_pts_list[i:i + args.batch_size]
        points_with_batch, batch_cnt = create_batch(
            batch_pts,
            force_fixed_n=args.force_fixed_n,
            fixed_n=args.fixed_n,
            seed=args.seed
        )

        cur_batch = base_dict.copy()
        cur_batch["points"] = points_with_batch
        cur_batch["batch_size"] = len(batch_pts)
        cur_batch["xyz_batch_cnt"] = batch_cnt
        load_data_to_gpu(cur_batch)

        with torch.no_grad():
            pred_dicts, _ = model(cur_batch)

        for pred in pred_dicts:
            new_scalar, hits, _ = get_target_scalar(
                pred,
                target_center=target_center,
                target_lwh=target_lwh,
                region_mode=region_mode if region_mode is not None else "box",
                region_expand=args.region_expand,
                region_radius=args.region_radius,
                class_filter=class_filter
            )
            drop = max(0.0, float(orig_scalar) - float(new_scalar))
            drops.append(drop)
            hit_counts.append(int(hits))

    drops = np.asarray(drops, dtype=np.float32)
    hit_counts = np.asarray(hit_counts, dtype=np.int32)

    m = min(len(drops), len(mask_info))
    for drop, (mode, mask) in zip(drops[:m], mask_info[:m]):
        if mode == "del":
            importance[mask] += float(drop)
        else:
            kept = max(0.0, float(orig_scalar) - float(drop))
            importance[mask] += float(kept)

    if importance.max() > 0:
        importance /= (importance.max() + 1e-12)

    diag = {
        "orig_scalar": float(orig_scalar),
        "orig_region_hits": int(orig_hits),
        "target_region": args.target_region,
        "region_expand": float(args.region_expand),
        "region_radius": float(args.region_radius),
        "target_class": args.target_class,
        "drops_min": float(drops.min()) if drops.size else 0.0,
        "drops_mean": float(drops.mean()) if drops.size else 0.0,
        "drops_max": float(drops.max()) if drops.size else 0.0,
        "region_hits_mean": float(hit_counts.mean()) if hit_counts.size else 0.0,
        "region_hits_zero_frac": float((hit_counts == 0).mean()) if hit_counts.size else 1.0,
    }
    logger.info(f"[DIAG] drops: min={diag['drops_min']:.6f} mean={diag['drops_mean']:.6f} max={diag['drops_max']:.6f}")
    logger.info(f"[DIAG] region_hits: mean={diag['region_hits_mean']:.3f} zero_frac={diag['region_hits_zero_frac']:.3f}")

    np.savez_compressed(
        out_root / "typeA_explanation.npz",
        points=points_xyz.astype(np.float32),
        points_full=points_np.astype(np.float32),
        importance=importance.astype(np.float32),
        drops=drops.astype(np.float32),
        region_hits=hit_counts.astype(np.int32),
        snapped_center=np.array(snapped_center, dtype=np.float32) if snapped_center is not None else np.zeros((3,), dtype=np.float32),
        snapped_lwh=np.array(snapped_lwh, dtype=np.float32) if snapped_lwh is not None else np.zeros((3,), dtype=np.float32),
        diag=np.string_(str(diag))
    )
    logger.info(f"[OK] Saved Type-A → {out_root}/typeA_explanation.npz")

    if len(scores) > 0:
        order = np.argsort(-scores)
        logger.info(f"[ALL-BOX] Explaining ALL {len(order)} detections (sorted by score descending)")

        for rank_idx, det_idx in enumerate(order, 1):
            score = float(scores[det_idx])
            label = int(labels[det_idx])
            box = boxes[det_idx].astype(np.float32)
            cls_name = cfg.CLASS_NAMES[label - 1] if 1 <= label <= len(cfg.CLASS_NAMES) else f"cls{label}"

            logger.info(f"[ALL-BOX] Rank {rank_idx}/{len(order)} - {cls_name} score={score:.6f} (idx={det_idx})")

            importance = np.zeros(len(points_xyz), dtype=np.float32)
            center = box[:3]
            size = box[3:6]

            expand = 1.6
            step_grid = 0.4
            xs = np.arange(center[0] - size[0] * expand, center[0] + size[0] * expand + 1e-6, step_grid)
            ys = np.arange(center[1] - size[1] * expand, center[1] + size[1] * expand + 1e-6, step_grid)
            zs = np.arange(center[2] - size[2] * expand, center[2] + size[2] * 1.5 + 1e-6, step_grid)

            if xs.size == 0 or ys.size == 0 or zs.size == 0:
                logger.warning(f"[SKIP] Empty grid for det{det_idx}")
                continue

            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            grid_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            np.random.shuffle(grid_centers)
            centers_per_scale = grid_centers[:max(1, args.num_masks // max(1, len(args.mask_scales)))]

            batch_pts_list = []
            mask_info = []

            t0 = time.time()
            for scale in args.mask_scales:
                half = float(scale) / 2.0
                for c in centers_per_scale:
                    lo = c - half
                    hi = c + half
                    inside = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)

                    masked_del = points_np[~inside]
                    if masked_del.shape[0] > 10:
                        batch_pts_list.append(masked_del)
                        mask_info.append(("del", inside))

                    if args.use_insertion:
                        masked_ins = points_np[inside]
                        if masked_ins.shape[0] > 10:
                            batch_pts_list.append(masked_ins)
                            mask_info.append(("ins", inside))

            logger.info(f"[ALL-BOX] Built {len(batch_pts_list)} masks for det{det_idx} in {time.time()-t0:.2f}s")
            if len(batch_pts_list) == 0:
                continue

            drops = []
            for i in tqdm(range(0, len(batch_pts_list), args.batch_size),
                          desc=f"Rank{rank_idx} {cls_name}", leave=False):
                batch_pts = batch_pts_list[i:i + args.batch_size]
                points_with_batch, batch_cnt = create_batch(
                    batch_pts,
                    force_fixed_n=args.force_fixed_n,
                    fixed_n=args.fixed_n,
                    seed=args.seed
                )

                cur_batch = base_dict.copy()
                cur_batch["points"] = points_with_batch
                cur_batch["batch_size"] = len(batch_pts)
                cur_batch["xyz_batch_cnt"] = batch_cnt
                load_data_to_gpu(cur_batch)

                with torch.no_grad():
                    pred_dicts, _ = model(cur_batch)

                for pred in pred_dicts:
                    new_scalar, _, _ = get_target_scalar(
                        pred,
                        target_center=center,
                        target_lwh=size,
                        region_mode="box",
                        region_expand=1.5,
                        class_filter=class_filter
                    )
                    drop = max(0.0, float(score) - float(new_scalar))
                    drops.append(drop)

            drops = np.asarray(drops, dtype=np.float32)

            m = min(len(drops), len(mask_info))
            for drop, (mode, mask) in zip(drops[:m], mask_info[:m]):
                if mode == "del":
                    importance[mask] += float(drop)
                else:
                    kept = max(0.0, float(score) - float(drop))
                    importance[mask] += float(kept)

            if importance.max() > 0:
                importance /= (importance.max() + 1e-12)

            save_dir = out_root / f"det{det_idx:03d}_{cls_name}_rank{rank_idx}"
            save_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                save_dir / "typeA_explanation.npz",
                points=points_xyz.astype(np.float32),
                points_full=points_np.astype(np.float32),
                importance=importance.astype(np.float32),
                drops=drops.astype(np.float32),
                box=box.astype(np.float32),
                score=np.float32(score),
                label=np.int32(label),
                class_name=np.string_(cls_name)
            )

            logger.info(f"[OK] Saved ALL-box Type-A → {save_dir}/typeA_explanation.npz")

    logger.info(f"========== DONE (Type A) Results in {out_root} ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        logger.error(traceback.format_exc())
        raise
