#!/usr/bin/env python3
"""
sonar_lime_typeA_objectness_DEBUG.py

Sonar LIME (Type-A objectness target) for PointRCNN (OpenPCDet) + DEBUG prints.

- Loads reconstructed SonarCloud .npy (N,3)
- Applies scale/shift + intensity (KITTI-like input)
- Clusters into superpoints (KMeans with FPS init)
- Perturbs by removing random clusters
- Target scalar = best detection score (global or restricted to snapped region)
- Fits Ridge regression for cluster importance
- Saves NPZ + JSON for visualization/debug
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def setup_logger():
    logger = logging.getLogger("LIME_SONAR_TYPEA_DEBUG")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not logger.hasHandlers():
        logger.addHandler(h)
    return logger


logger = setup_logger()


def fps_init_centers(xyz: np.ndarray, n: int) -> np.ndarray:
    """Returns centers (n,3). Always 2D even if n=1."""
    import torch as _torch
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"fps_init_centers expects (N,3), got {xyz.shape}")

    n = int(min(n, xyz.shape[0]))
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = _torch.from_numpy(xyz).float().cuda().unsqueeze(0)
    idx_tensor, = pointnet2_utils.FarthestPointSampling.apply(
        pts, _torch.tensor([n], device=pts.device)
    )

    centers = pts[0][idx_tensor[0]].detach().cpu().numpy()
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, 3)
    return centers


def sonar_transform(points_xyz: np.ndarray,
                    xyz_scale: float,
                    xyz_shift,
                    intensity_mode: str,
                    intensity_value: float,
                    seed: int) -> np.ndarray:
    """Transform sonar xyz (N,3) -> (N,4) by scale/shift + synthetic intensity."""
    xyz = np.asarray(points_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected sonar xyz shape (N,3), got {xyz.shape}")

    xyz = xyz * float(xyz_scale)
    xyz = xyz + np.array(xyz_shift, dtype=np.float32)

    rng = np.random.default_rng(seed)
    if intensity_mode == "random":
        inten = rng.uniform(0.1, 0.9, (xyz.shape[0], 1)).astype(np.float32)
    elif intensity_mode == "uniform_high":
        inten = np.full((xyz.shape[0], 1), 0.7, dtype=np.float32)
    else:
        inten = np.full((xyz.shape[0], 1), float(intensity_value), dtype=np.float32)

    pts4 = np.hstack([xyz, inten]).astype(np.float32)

    finite = np.isfinite(pts4).all(axis=1)
    pts4 = pts4[finite]
    nonzero = np.linalg.norm(pts4[:, :3], axis=1) > 1e-6
    pts4 = pts4[nonzero]

    return pts4


def to_fixed_n(pts4: np.ndarray, fixed_n: int, seed: int) -> np.ndarray:
    """Deterministic pad/downsample to fixed_n points."""
    rng = np.random.default_rng(seed)
    pts4 = np.asarray(pts4, dtype=np.float32)
    N = pts4.shape[0]
    if N == 0:
        return np.zeros((fixed_n, 4), dtype=np.float32)

    if N < fixed_n:
        dup = pts4[rng.choice(N, fixed_n - N, replace=True)]
        return np.concatenate([pts4, dup], axis=0).astype(np.float32)
    if N > fixed_n:
        idx = rng.choice(N, fixed_n, replace=False)
        return pts4[idx].astype(np.float32)
    return pts4.astype(np.float32)


def batch_to_pcdet(points_list):
    """
    points_list: list of (Ni,4) arrays
    returns:
      points: (sum_i Ni, 5) with batch_idx prepended (OpenPCDet format)
      xyz_batch_cnt: (B,) int32
    """
    all_pts, cnts = [], []
    for b, pts in enumerate(points_list):
        pts = np.asarray(pts, dtype=np.float32)
        cnts.append(int(pts.shape[0]))
        bidx = np.full((pts.shape[0], 1), b, dtype=np.float32)
        all_pts.append(np.hstack([bidx, pts]).astype(np.float32))

    points = torch.from_numpy(np.concatenate(all_pts, axis=0)).float().cuda()
    xyz_batch_cnt = torch.tensor(cnts, dtype=torch.int32, device="cuda")
    return points, xyz_batch_cnt


def get_target_scalar(pred_dict, target_center=None, target_lwh=None,
                      region_mode="box", region_expand=1.3, region_radius=6.0,
                      class_filter=None):
    """Return (best_score, hits, best_index) under optional region/class filters."""
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

    best = idxs[np.argmax(scores[idxs])]
    return float(scores[best]), int(idxs.size), int(best)


def load_snapped_bounds(recon_csv: Path, sample_name: str,
                        xyz_scale: float, xyz_shift):
    """
    Load object bounds from recon_summary.csv (min/max in ORIGINAL coords),
    then map to model-space using scale+shift.
    """
    import pandas as pd
    df = pd.read_csv(recon_csv)
    row = df[df["name"] == sample_name]
    if len(row) == 0:
        return None, None

    obj_min = row[["min_x", "min_y", "min_z"]].values[0].astype(np.float32)
    obj_max = row[["max_x", "max_y", "max_z"]].values[0].astype(np.float32)

    obj_min = obj_min * float(xyz_scale) + np.array(xyz_shift, dtype=np.float32)
    obj_max = obj_max * float(xyz_scale) + np.array(xyz_shift, dtype=np.float32)

    center = (obj_min + obj_max) / 2.0
    lwh = (obj_max - obj_min)
    return center.astype(np.float32), lwh.astype(np.float32)


def print_top10_preds(pred_dict, class_names, title="[TOP10]"):
    scores_t = pred_dict.get("pred_scores", None)
    boxes_t = pred_dict.get("pred_boxes", None)
    labels_t = pred_dict.get("pred_labels", None)

    if scores_t is None or boxes_t is None or labels_t is None:
        logger.info(f"{title} pred dict missing keys")
        return

    scores = scores_t.detach().cpu().numpy()
    boxes = boxes_t.detach().cpu().numpy()
    labels = labels_t.detach().cpu().numpy()

    logger.info(f"{title} total_preds={len(scores)}")
    if len(scores) == 0:
        return

    order = np.argsort(-scores)[:10]
    for r, i in enumerate(order, 1):
        b = boxes[i]
        lab = int(labels[i])
        cls = class_names[lab - 1] if 1 <= lab <= len(class_names) else f"cls{lab}"
        logger.info(
            f"{title} {r:02d}: score={scores[i]:.6f} cls={cls} "
            f"center=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f}) "
            f"lwh=({b[3]:.2f},{b[4]:.2f},{b[5]:.2f}) yaw={b[6]:.2f}"
        )


def main():
    ap = argparse.ArgumentParser("Sonar LIME Type-A objectness (DEBUG)")

    ap.add_argument("--cfg", required=True)        # TODO(PATH): model config YAML (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)
    ap.add_argument("--ckpt", required=True)       # TODO(PATH): checkpoint path (e.g., checkpoints/pointrcnn_7870.pth)
    ap.add_argument("--root_path", required=True)  # TODO(PATH): KITTI root path required by OpenPCDet infra
    ap.add_argument("--info_dir", required=True)   # TODO(PATH): directory containing kitti_infos_*.pkl (and dbinfos)

    ap.add_argument("--sonar_npy", required=True, help="reconstructed .npy from reconstruct_batch.py")  # TODO(PATH)
    ap.add_argument("--recon_summary_csv", required=True)  # TODO(PATH): recon_summary.csv
    ap.add_argument("--sample_name", default=None)

    ap.add_argument("--xyz_scale", type=float, default=5.0)
    ap.add_argument("--xyz_shift", type=float, nargs=3, default=[25.0, 0.0, -1.5])
    ap.add_argument("--intensity_mode", choices=["const", "random", "uniform_high"], default="uniform_high")
    ap.add_argument("--sonar_intensity", type=float, default=0.5)

    ap.add_argument("--target_region", choices=["global", "box", "radius"], default="box")
    ap.add_argument("--region_expand", type=float, default=1.3)
    ap.add_argument("--region_radius", type=float, default=6.0)
    ap.add_argument("--target_class", type=str, default="any")

    ap.add_argument("--num_clusters", type=int, default=256)
    ap.add_argument("--num_perturbs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--min_keep_frac", type=float, default=0.4)
    ap.add_argument("--fixed_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=str, default="lime_sonar_results")  # TODO(PATH): output directory
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sample_name = args.sample_name or Path(args.sonar_npy).stem

    cfg_from_yaml_file(args.cfg, cfg)

    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 1e-6
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
    logger.info("[OK] model loaded (eval).")
    logger.info(f"[CFG] SCORE_THRESH forced to {cfg.MODEL.POST_PROCESSING.SCORE_THRESH}")

    pts_xyz_raw = np.load(args.sonar_npy).astype(np.float32)
    pts4 = sonar_transform(
        pts_xyz_raw,
        xyz_scale=args.xyz_scale,
        xyz_shift=args.xyz_shift,
        intensity_mode=args.intensity_mode,
        intensity_value=args.sonar_intensity,
        seed=args.seed
    )
    pts_xyz = pts4[:, :3].copy()
    logger.info(f"[DATA] points after transform: {pts4.shape}")

    snapped_center, snapped_lwh = load_snapped_bounds(
        Path(args.recon_summary_csv), sample_name,
        xyz_scale=args.xyz_scale, xyz_shift=args.xyz_shift
    )

    if args.target_region == "global" or snapped_center is None:
        target_center, target_lwh = None, None
        region_mode = "box"
        logger.info("[TARGET] GLOBAL objectness target (no region filter)")
    else:
        target_center, target_lwh = snapped_center, snapped_lwh
        region_mode = "radius" if args.target_region == "radius" else "box"
        logger.info(f"[TARGET] region={region_mode} center={target_center.tolist()} lwh={target_lwh.tolist()}")
        logger.info(f"[TARGET] region_expand={args.region_expand} region_radius={args.region_radius}")

    class_filter = None
    if args.target_class.lower() != "any":
        name_to_id = {n.lower(): (i + 1) for i, n in enumerate(cfg.CLASS_NAMES)}
        if args.target_class.lower() not in name_to_id:
            raise ValueError(f"--target_class {args.target_class} not in {cfg.CLASS_NAMES}")
        class_filter = name_to_id[args.target_class.lower()]
        logger.info(f"[TARGET] class_filter={args.target_class} -> id={class_filter}")
    else:
        logger.info("[TARGET] class_filter=ANY")

    base_pts = to_fixed_n(pts4, args.fixed_n, seed=args.seed)
    points_t, cnt_t = batch_to_pcdet([base_pts])

    batch = {"frame_id": sample_name, "points": points_t, "batch_size": 1, "xyz_batch_cnt": cnt_t}
    load_data_to_gpu(batch)

    with torch.no_grad():
        pred_dicts, _ = model(batch)
    base_pred = pred_dicts[0]

    print_top10_preds(base_pred, cfg.CLASS_NAMES, title="[GLOBAL TOP10]")

    base_scalar, base_hits, base_best = get_target_scalar(
        base_pred,
        target_center=target_center,
        target_lwh=target_lwh,
        region_mode=region_mode,
        region_expand=args.region_expand,
        region_radius=args.region_radius,
        class_filter=class_filter
    )
    logger.info(f"[BASE TARGET] scalar={base_scalar:.6f} hits={base_hits} best_idx={base_best}")

    if args.target_region != "global" and base_hits == 0:
        logger.warning(
            "[WARN] Region hits are ZERO on baseline. Model has no boxes inside snapped region.\n"
            "Try: --target_region global OR increase --region_expand / --region_radius."
        )

    N = pts_xyz.shape[0]
    K = int(min(args.num_clusters, N))
    if K < 2:
        raise RuntimeError(f"Not enough points ({N}) to cluster into K={K}")

    init_centers = fps_init_centers(pts_xyz, K)

    if init_centers.ndim != 2 or init_centers.shape[1] != 3:
        raise RuntimeError(f"Bad init_centers shape: {init_centers.shape}")
    if init_centers.shape[0] < K:
        extra = pts_xyz[np.random.choice(N, K - init_centers.shape[0], replace=False)]
        init_centers = np.vstack([init_centers, extra])
    init_centers = init_centers[:K].reshape(K, 3).astype(np.float32)

    logger.info(f"[CLUST] Running KMeans: N={N} K={K} init_centers={init_centers.shape}")

    km = KMeans(n_clusters=K, init=init_centers, n_init=1, max_iter=50, random_state=args.seed)
    cluster_labels = km.fit_predict(pts_xyz).astype(np.int32)

    cluster_pts = [pts4[cluster_labels == i] for i in range(K)]
    sizes = np.array([c.shape[0] for c in cluster_pts], dtype=np.int32)
    logger.info(f"[CLUST] cluster size stats: min={sizes.min()} mean={sizes.mean():.1f} max={sizes.max()}")

    min_keep = max(1, int(args.min_keep_frac * K))
    masks = []
    rng = np.random.default_rng(args.seed)
    for _ in range(args.num_perturbs):
        keep_k = rng.integers(min_keep, K + 1)
        keep = rng.choice(K, keep_k, replace=False)
        masks.append(keep)

    logger.info(f"[PERT] num_perturbs={args.num_perturbs} min_keep={min_keep}/{K} batch_size={args.batch_size}")

    y = np.zeros(args.num_perturbs, dtype=np.float32)
    region_hits = np.zeros(args.num_perturbs, dtype=np.int32)

    for start in range(0, args.num_perturbs, args.batch_size):
        end = min(start + args.batch_size, args.num_perturbs)

        clouds = []
        for i in range(start, end):
            keep = masks[i]
            pts_keep = np.vstack([cluster_pts[j] for j in keep if cluster_pts[j].shape[0] > 0]).astype(np.float32)
            pts_keep = to_fixed_n(pts_keep, args.fixed_n, seed=args.seed + i)
            clouds.append(pts_keep)

        points_t, cnt_t = batch_to_pcdet(clouds)
        bdict = {"frame_id": sample_name, "points": points_t, "batch_size": len(clouds), "xyz_batch_cnt": cnt_t}
        load_data_to_gpu(bdict)

        with torch.no_grad():
            preds, _ = model(bdict)

        for local_i, global_i in enumerate(range(start, end)):
            pred = preds[local_i]
            s, hits, _ = get_target_scalar(
                pred,
                target_center=target_center,
                target_lwh=target_lwh,
                region_mode=region_mode,
                region_expand=args.region_expand,
                region_radius=args.region_radius,
                class_filter=class_filter
            )
            y[global_i] = s
            region_hits[global_i] = hits

    logger.info(f"[Y] score stats: min={y.min():.6f} mean={y.mean():.6f} max={y.max():.6f}")
    logger.info(f"[Y] region_hits: mean={region_hits.mean():.3f} zero_frac={(region_hits==0).mean():.3f}")

    if float(y.max()) <= 1e-9:
        logger.warning(
            "[WARN] All perturbation target scores are ~0. LIME cannot learn meaningful weights.\n"
            "Try: --target_region global OR larger --region_expand / --region_radius."
        )

    X = np.zeros((args.num_perturbs, K), dtype=np.float32)
    for i, keep in enumerate(masks):
        X[i, keep] = 1.0

    ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=args.seed)
    ridge.fit(X, y)

    w = ridge.coef_.astype(np.float32)
    b = float(ridge.intercept_)
    r2 = float(ridge.score(X, y))
    logger.info(f"[LIME] R2={r2:.4f} intercept={b:.6f}")

    point_importance_raw = w[cluster_labels].astype(np.float32)
    denom = float(np.max(np.abs(point_importance_raw))) if point_importance_raw.size else 1.0
    point_importance = (point_importance_raw / (denom + 1e-12)).astype(np.float32)

    out_root = Path(args.out_dir) / sample_name
    out_root.mkdir(parents=True, exist_ok=True)

    npz_path = out_root / "lime_typeA_objectness.npz"
    np.savez_compressed(
        npz_path,
        points=pts_xyz.astype(np.float32),
        points_full=pts4.astype(np.float32),
        cluster_labels=cluster_labels.astype(np.int32),
        num_clusters=np.int32(K),
        point_importance=point_importance.astype(np.float32),
        raw_point_importance=point_importance_raw.astype(np.float32),
        cluster_weights=w.astype(np.float32),
        intercept=np.float32(b),
        r2=np.float32(r2),
        base_scalar=np.float32(base_scalar),
        base_hits=np.int32(base_hits),
        y=y.astype(np.float32),
        region_hits=region_hits.astype(np.int32),
        target_region=np.bytes_(args.target_region),
        target_class=np.bytes_(args.target_class),
        region_expand=np.float32(args.region_expand),
        region_radius=np.float32(args.region_radius),
        snapped_center=np.array(snapped_center, dtype=np.float32) if snapped_center is not None else np.zeros((3,), dtype=np.float32),
        snapped_lwh=np.array(snapped_lwh, dtype=np.float32) if snapped_lwh is not None else np.zeros((3,), dtype=np.float32),
    )

    meta = {
        "sample": sample_name,
        "sonar_npy": str(Path(args.sonar_npy).resolve()),
        "recon_summary_csv": str(Path(args.recon_summary_csv).resolve()),
        "num_points_after_transform": int(pts_xyz.shape[0]),
        "num_clusters": int(K),
        "num_perturbs": int(args.num_perturbs),
        "fixed_n": int(args.fixed_n),
        "score_thresh_forced": float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH),
        "base_scalar": float(base_scalar),
        "base_hits": int(base_hits),
        "y_min": float(y.min()),
        "y_mean": float(y.mean()),
        "y_max": float(y.max()),
        "region_hits_zero_frac": float((region_hits == 0).mean()),
        "r2": float(r2),
        "intercept": float(b),
        "target_region": args.target_region,
        "target_class": args.target_class,
        "region_expand": float(args.region_expand),
        "region_radius": float(args.region_radius),
        "snapped_center": snapped_center.tolist() if snapped_center is not None else None,
        "snapped_lwh": snapped_lwh.tolist() if snapped_lwh is not None else None,
    }
    with open(out_root / "lime_typeA_objectness.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[OK] Saved NPZ:  {npz_path}")
    logger.info(f"[OK] Saved JSON: {out_root/'lime_typeA_objectness.json'}")

    if args.target_region != "global" and base_hits == 0:
        logger.info(
            "\n[RECOMMENDATION]\n"
            "Baseline had zero region hits. Try:\n"
            "  1) --target_region global\n"
            "  2) bigger region: --region_expand 2.5 or 3.0\n"
            "  3) radius mode: --target_region radius --region_radius 10\n"
        )


if __name__ == "__main__":
    main()
