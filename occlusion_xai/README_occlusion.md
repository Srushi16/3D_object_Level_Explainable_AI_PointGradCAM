Occlusion / Point Masking Explainer (OpenPCDet / PointRCNN)
=========================================================

This script implements a SOTA-style point masking (occlusion) explainer for
3D object detection on point clouds using PointRCNN (OpenPCDet).

For each detected object in a KITTI frame (above --score_thr), the script
creates many masked variants of the point cloud and observes how the detector’s
confidence changes. The aggregated confidence drops are converted into a
per-point importance (saliency) map.

In addition to generating explanation files, the script computes
LIME / Grad-CAM–aligned object-level metrics.


------------------------------------------------------------
1) Metrics computed
------------------------------------------------------------

Focus Ratio (higher is better):
    Fraction of total importance that lies inside an expanded 3D bounding box
    around the detected object.

Sparsity@alpha on support (higher is better):
    Fraction of importance mass contained in the top-alpha fraction of points.
    Computed only on non-zero saliency points to avoid artificial sparsity
    inflation.

Faithfulness ΔScore@alpha (higher is better):
    Drop in matched detection confidence after removing the top-alpha most
    important points, measured using strict box + class matching against
    a baseline evaluation score computed on the same evaluation subset.

Deletion Curve + AUC (lower is better):
    Remaining matched confidence as increasingly important points are removed.
    The area under this curve summarizes deletion faithfulness.


------------------------------------------------------------
2) Required inputs
------------------------------------------------------------

You must provide the following paths and arguments:

--cfg
    OpenPCDet model configuration YAML
    (e.g., tools/cfgs/kitti_models/pointrcnn.yaml)

--ckpt
    Trained model checkpoint (.pth)

--root_path
    KITTI dataset root directory

--info_dir
    Directory containing KITTI info files used by KittiZipDataset
    (e.g., kitti_infos_train.pkl, kitti_infos_val.pkl, ...)

--sample_idx
    Frame index to explain (integer, e.g., 62)


------------------------------------------------------------
3) Outputs
------------------------------------------------------------

All outputs are written under:

    --out/sample_XXXXXX/<ClassName>/detYYY/

Per detection:
    explanation.npz

Contents of explanation.npz:
    points        : (N,3) original XYZ points (for visualization)
    importance    : (N,) per-point importance scores
    box           : predicted 3D bounding box
    score         : predicted confidence score
    label         : predicted class label
    eval_idx      : indices of points used for metric evaluation
    focus_ratio
    sparsity
    baseline_eval
    delta_s_alpha
    deletion_fracs
    deletion_scores
    deletion_auc

Per sample:
    metrics_summary.csv
        One row per explained detection, used for aggregated plots
        and cross-method comparison.


------------------------------------------------------------
4) How to run
------------------------------------------------------------

Example command:

    python occlusion_metrics.py \
        --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
        --ckpt checkpoints/pointrcnn_7870.pth \
        --root_path /path/to/KITTI \
        --info_dir /path/to/KITTI \
        --sample_idx 62 \
        --out occlusion_out


------------------------------------------------------------
5) Paths you must update
------------------------------------------------------------

--cfg        : path to PointRCNN YAML configuration
--ckpt       : path to trained checkpoint (.pth)
--root_path  : KITTI dataset root directory
--info_dir   : directory containing KITTI info files
--out        : output directory for explanations and metrics


------------------------------------------------------------
6) Notes / common pitfalls
------------------------------------------------------------

- This script requires a working CUDA GPU environment.
  torch.cuda.is_available() must return True.

- If strict IoU + class matching fails during evaluation, the matched
  confidence becomes 0.0. This will appear as:
      [WARN baseline_eval=0]
  in the printed metric logs.

- Determinism:
  A fixed SEED=42 is used for:
      - occlusion grid shuffling
      - point padding / downsampling
      - evaluation subset selection
  This ensures metrics are reproducible across runs.

You are now ready to generate occlusion-based explanations and metrics.
