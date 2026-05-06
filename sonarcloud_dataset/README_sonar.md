# SONAR LIME (Objectness) for PointRCNN (OpenPCDet)

This script generates a **LIME explanation** on SONAR reconstructed point clouds by:
- loading a reconstructed SONAR `.npy` (N,3),
- applying KITTI-like transform (scale/shift + synthetic intensity),
- clustering points into superpoints (KMeans with FPS init),
- perturbing by removing random superpoints,
- using a SONAR-robust target: **best detection score** (global or restricted to a snapped region),
- fitting Ridge regression to get cluster weights,
- exporting a single `.npz` for visualization + a `.json` metadata file.

## Inputs (required)
- `--cfg` : model config YAML (PointRCNN)
- `--ckpt`: trained checkpoint
- `--root_path`: KITTI root path (required by OpenPCDet dataset infra)
- `--info_dir`: directory containing `kitti_infos_*.pkl` (and dbinfos)
- `--sonar_npy`: reconstructed SONAR point cloud `.npy` (N,3)
- `--recon_summary_csv`: CSV containing snapped bounds for the object (`min_x..max_z`)

## Output
Written under:
`--out_dir/<sample_name>/`

- `lime_typeA_objectness.npz`
- `lime_typeA_objectness.json`

The NPZ contains:
- `points` (N,3), `points_full` (N,4)
- `point_importance` (N,), `cluster_labels`, `cluster_weights`
- diagnostics: `base_scalar`, `base_hits`, `region_hits`, `r2`, etc.

## Target region modes
- `--target_region global`: use best score anywhere (recommended sanity-check)
- `--target_region box`: best score inside snapped 3D box (with `--region_expand`)
- `--target_region radius`: best score within radius of snapped center

## YAML files needed
- **Model YAML**: e.g. `tools/cfgs/kitti_models/pointrcnn.yaml`
- **Dataset YAML**: `tools/cfgs/dataset_configs/kitti_dataset.yaml`  
  (used internally by OpenPCDet; this script overrides `DATA_PATH` and `INFO_DIR` via args)

## Example run
```bash
python sonar_lime.py \
  --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
  --ckpt checkpoints/pointrcnn_7870.pth \
  --root_path /dataset/original \
  --info_dir /dataset/Kitti_infos \
  --sonar_npy data/sonarcloud/recon_out/boat_noTerrain_ori1.npy \
  --recon_summary_csv data/sonarcloud/recon_out/recon_summary.csv \
  --target_region box --region_expand 1.3 \
  --fixed_n 10000 \
  --out_dir lime_sonar_results


# Sonar Occlusion (OpenPCDet / PointRCNN)

This script computes ** occlusion explanations** on SONAR point clouds:
“How much does the model’s best detection score drop when we remove points in a local region?”

is designed for **domain shift** cases where box tracking / IoU matching is unreliable.

## Inputs (required)
- `--cfg` : PointRCNN model YAML (KITTI model config)
- `--ckpt`: trained checkpoint
- `--root_path`, `--info_dir`: KITTI metadata paths (OpenPCDet needs dataset metadata even for SONAR)
- `--sonar_npy`: SONAR point cloud `.npy` (expects Nx3 or Nx4)

Optional but recommended:
- `--recon_summary_csv`: CSV with per-sample object bounds (`min_x,min_y,min_z,max_x,max_y,max_z`)
  used to define a stable “target region” around the object.

## SONAR .npy format
- `float32` array with shape:
  - `(N,3)` = x,y,z  OR
  - `(N,4)` = x,y,z,intensity
If intensity is missing, script creates it using `--intensity_mode`.

## Output
Creates:
`<out>/<sample_tag>/`
- `orig_pred.npz` (raw predictions + points)
- `raw_detection_top10.csv`
- `typeA_explanation.npz` (global/snap-based explanation)
- For ALL predicted detections:
  `<out>/<sample_tag>/detXXX_<Class>_rankR/typeA_explanation.npz`

## Run (example)
```bash
python occlusion.py \
  --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
  --ckpt checkpoints/pointrcnn_7870.pth \
  --root_path /path/to/KITTI \
  --info_dir /path/to/KITTI/Kitti_infos \
  --sonar_npy data/sonarcloud/recon_out/boat_noTerrain_ori1.npy \
  --recon_summary_csv data/sonarcloud/recon_out/recon_summary.csv \
  --target_region box --region_expand 1.3 \
  --fixed_n 10000 --force_fixed_n \
  --out occlusion_sonar_results

PointGrad-CAM for SONAR Dataset (OpenPCDet / PointRCNN)
=====================================================

This script runs PointGrad-CAM on SONAR point cloud data using a PointRCNN model
from OpenPCDet. It adapts a KITTI-trained detector to SONAR data by applying
geometric transformations and generates point-level saliency maps explaining
the model’s predictions.

The script supports two explanation modes:
- Objectness-based Grad-CAM (default, recommended for SONAR)
- Per-detection Grad-CAM (optional)


------------------------------------------------------------
1) What this script does
------------------------------------------------------------

- Loads a SONAR point cloud from a .npy file (N,3)
- Applies scale and shift to align SONAR coordinates with KITTI space
- Adds artificial intensity to form (N,4) points
- Pads or downsamples to a fixed number of points
- Runs PointRCNN forward pass
- Captures intermediate ROI features and classification logits
- Computes Grad-CAM saliency
- Maps pooled saliency back to original points
- Exports colored point clouds and metadata

This script is designed for explainability and analysis, not training.


------------------------------------------------------------
2) Supported modes
------------------------------------------------------------

Mode A: Objectness Grad-CAM (default)
- Explains the strongest ROI logit in the scene
- Produces ONE explanation per SONAR sample
- Recommended for SONAR data where detections may be unstable

Mode B: Per-detection Grad-CAM (--per_detect)
- Explains each final detection above --min_score
- Produces multiple explanations per sample
- Requires reliable post-processed detections


------------------------------------------------------------
3) Required inputs
------------------------------------------------------------

You must provide the following arguments:

--cfg
    OpenPCDet PointRCNN model YAML
    (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)

--ckpt
    Trained PointRCNN checkpoint (.pth)

--root_path
    KITTI dataset root directory
    (used only to build the model and load metadata)

--info_dir
    Directory containing KITTI info files
    (kitti_infos_train.pkl, kitti_infos_val.pkl, etc.)

--sonar_npy
    Path to SONAR point cloud (.npy), shape (N,3)

--outdir
    Output directory for Grad-CAM results


------------------------------------------------------------
4) SONAR-specific preprocessing
------------------------------------------------------------

The SONAR point cloud is transformed before inference:

- Scaling:
    xyz = xyz * --xyz_scale

- Shifting:
    xyz = xyz + --xyz_shift

- Intensity:
    A synthetic intensity channel is added:
      - uniform_high (default)
      - constant (--sonar_intensity)
      - random

- Fixed number of points:
    Points are padded or downsampled to --fixed_n
    to match PointRCNN input requirements.


------------------------------------------------------------
5) Outputs
------------------------------------------------------------

All outputs are written under:

    --outdir/<sample_name>/

Mode A (objectness):
    gradcam_objectness.ply
    gradcam_objectness.npz
    gradcam_objectness_debug.npz
    metadata.json

Mode B (per-detection):
    <ClassName>/detXXX_scoreY/
        heatmap.ply
        heatmap.npz
        debug.npz
    detections.csv
    metadata.json

heatmap.npz contains:
    points  : (N,3)
    scores  : (N,) normalized Grad-CAM importance
    colors  : (N,3) RGB colors for visualization


------------------------------------------------------------
6) Snapped region support (optional)
------------------------------------------------------------

If a reconstruction summary CSV is provided:

--recon_summary_csv <file.csv>

The script can restrict ROI selection to:
- global  (default)
- radius  (distance-based)
- box     (axis-aligned bounding region)

This helps focus explanations on a known object region
in noisy SONAR scenes.


------------------------------------------------------------
7) How to run
------------------------------------------------------------

Example (objectness mode):

    python sonar_run_gradcam.py \
        --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
        --ckpt checkpoints/pointrcnn_7870.pth \
        --root_path /path/to/KITTI \
        --info_dir /path/to/KITTI/Kitti_infos \
        --sonar_npy data/sonar/sample_01.npy \
        --outdir gradcam_sonar_results

Example (per-detection mode):

    python sonar_run_gradcam.py \
        --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
        --ckpt checkpoints/pointrcnn_7870.pth \
        --root_path /path/to/KITTI \
        --info_dir /path/to/KITTI/Kitti_infos \
        --sonar_npy data/sonar/sample_01.npy \
        --per_detect \
        --outdir gradcam_sonar_results


------------------------------------------------------------
8) Paths you must update
------------------------------------------------------------

--cfg           : PointRCNN model YAML
--ckpt          : trained checkpoint (.pth)
--root_path     : KITTI dataset root
--info_dir      : KITTI info files directory
--sonar_npy     : SONAR point cloud (.npy)
--outdir        : output directory


------------------------------------------------------------
9) Notes / common pitfalls
------------------------------------------------------------

- CUDA is required. The script assumes torch.cuda.is_available() is True.
- The model is trained on KITTI; SONAR adaptation relies entirely on
  scale/shift and intensity heuristics.
- If no ROIs are captured, the script falls back to global max logits.
- Grad-CAM quality depends strongly on:
    - chosen activation layer (--act-layer)
    - mapping mode (radius vs knn)
    - fixed_n and point density

This script is intended for research and explainability analysis,
not for quantitative SONAR detection performance.
