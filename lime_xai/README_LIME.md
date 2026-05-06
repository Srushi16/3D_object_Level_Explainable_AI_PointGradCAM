# Object-centric LIME for PointRCNN (OpenPCDet / KITTI)

This script generates **object-level LIME explanations** for PointRCNN detections on KITTI-style point clouds.
For each detected object (above your thresholds), it fits a local surrogate model by perturbing points and measuring
the detector’s score change, producing a **per-point importance (saliency)** map.

It outputs:
- per-detection explanation `.npz` files (saliency + metadata + evaluation indices)
- a per-sample `metrics_summary.csv` (one row per explained detection)

---

## What it produces

For a given `--sample_idx`, outputs are created under:

`<OUTPUT_ROOT>/sample_XXXXXX/`

Where:
- `OUTPUT_ROOT` is derived from `--out` (the script treats `--out` as an output path; the **directory of `--out`** is used as the root output folder).
  - Example: if `--out results/dummy.npz`, then `OUTPUT_ROOT = results/`

Generated files:

- `sample_XXXXXX/metrics_summary.csv`
- `sample_XXXXXX/sample_XXXXXX_metadata.json`

Per detection:
- `sample_XXXXXX/<ClassName>/detYYY/lime_sampleXXXXXX_detYYY.npz`

Multi-class combined (optional / if enabled by script):
- `sample_XXXXXX/lime_sampleXXXXXX_all_classes.npz`

---

## Per-detection NPZ contents

Each per-detection `.npz` typically contains:

**Core**
- `points` `(N,4)` : `(x,y,z,intensity)` float32
- `point_importance` `(N,)` : LIME importance per point (saliency)
- `cluster_labels` `(N,)` : superpoint/cluster assignment used by LIME (if clustering is used)
- `pred_box` `(7,)` : predicted 3D box (KITTI-style: center + dims + yaw)
- `pred_score` `(1,)`
- `pred_label` `(1,)`

**Metrics (aligned with your thesis metrics)**
- `focus_ratio`
- `sparsity`
- `baseline_eval`
- `delta_s_alpha`

**Deletion faithfulness**
- `deletion_fracs`
- `deletion_scores`
- `deletion_auc`

**Reproducibility**
- `eval_idx` : indices of points used for metric evaluation (deterministic)

> Note: exact key names depend on the script version, but the structure above is the intended contract.

---

## Required input format

- Point clouds must be KITTI-style LiDAR points: `float32 (N,4)` = `(x,y,z,intensity)`
- Dataset uses `KittiZipDataset` (OpenPCDet) and expects **pre-generated KITTI info files** in `--info_dir`
  (e.g., `kitti_infos_val.pkl`, `kitti_infos_test.pkl`, etc.), and the KITTI root at `--root_path`.

---

## YAML files needed

1) **Model YAML** (required to run the model)
   - Example: `tools/cfgs/kitti_models/pointrcnn.yaml`

2) **Dataset YAML** (usually referenced by the model config)
   - Example: `tools/cfgs/dataset_configs/kitti_dataset.yaml`
   - Must match your KITTI setup (paths / splits / info file references), unless your script overrides via `--root_path` and `--info_dir`.

---

## Run (example)

```bash
python lime_object_centric.py \
  --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
  --ckpt checkpoints/pointrcnn.pth \
  --root_path /path/to/KITTI \
  --info_dir /path/to/KITTI/Kitti_infos \
  --sample_idx 62 \
  --object_centric \
  --out /path/to/results/dummy.npz
