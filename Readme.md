OpenPCDet – Setup and KITTI Dataset Preparation Guide
====================================================

This document provides a step-by-step guide to install OpenPCDet, prepare the
KITTI dataset, generate required info files, and run a quick evaluation to
verify that everything works correctly.

This guide is intended for first-time OpenPCDet users.


----------------------------------------------------
1. Environment Setup
----------------------------------------------------

Use Requirements.txt file tp install all dependancies. 
If you are using Container make sure to install Requirements.txt everytime you use container. 

----------------------------------------------------
2. Install OpenPCDet
----------------------------------------------------

Install this only Once. 

From the OpenPCDet root directory:

    python setup.py develop

Verify installation:

    python -c "import pcdet; print('OpenPCDet installed')"


----------------------------------------------------
3. KITTI Dataset Preparation
----------------------------------------------------

3.1 Expected KITTI directory structure

Option A – Pre-generated info files (recommended):

    <KITTI_ROOT>/
        training/
            velodyne/
            label_2/
            calib/
            image_2/
        testing/
            velodyne/
            calib/
            image_2/
        ImageSets/
            train.txt
            val.txt
            test.txt
        Kitti_infos/
            kitti_infos_train.pkl
            kitti_infos_val.pkl
            kitti_infos_test.pkl
            kitti_dbinfos_train.pkl
            gt_database/


Option B – ZIP-based KITTI (fallback supported):

    <KITTI_ROOT>/
        original/
            data_object_velodyne.zip
            data_object_label_2.zip
            data_object_calib.zip
            data_object_image_2.zip

The custom KittiZipDataset can automatically fall back to ZIP files
if info files are missing.


----------------------------------------------------
4. Generate KITTI Info Files
----------------------------------------------------

4.1 Dataset YAML configuration

Ensure the dataset YAML is correctly set:

    tools/cfgs/dataset_configs/kitti_dataset.yaml

Important fields:
- DATA_PATH   : path to KITTI root
- INFO_DIR    : directory where info files will be saved
- DATA_SPLIT  : train / val / test mapping


4.2 Generate info files (ZIP or folder based)

From the OpenPCDet root:

    python pcdet/datasets/kitti/kitti_zip_dataset.py create_kitti_zip_infos \
        tools/cfgs/dataset_configs/kitti_dataset.yaml

This generates:

    kitti_infos_train.pkl
    kitti_infos_val.pkl
    kitti_infos_test.pkl
    kitti_infos_trainval.pkl
    kitti_dbinfos_train.pkl
    gt_database/

These files are mandatory for training, evaluation, and explainability.


----------------------------------------------------
5. Verify Dataset Setup (Quick Evaluation)
----------------------------------------------------

Before running custom code or XAI pipelines, verify that OpenPCDet can
load the dataset and run inference.


5.1 Model configuration

Example model config:

    tools/cfgs/kitti_models/pointrcnn.yaml

Ensure:
- DATA_CONFIG.DATA_PATH matches your KITTI root
- DATA_CONFIG.INFO_PATH points to generated .pkl files


5.2 Run evaluation on validation set

    python tools/test.py \
        --cfg tools/cfgs/kitti_models/pointrcnn.yaml \
        --ckpt checkpoints/pointrcnn.pth \
        --eval_all

Expected result:
- KITTI AP metrics printed to console
- No dataset, calibration, or loading errors

If this works, your dataset setup is correct.


----------------------------------------------------
6. Common Issues and Tips
----------------------------------------------------

CUDA / PyTorch mismatch:
- Most runtime errors come from incompatible CUDA, PyTorch, or spconv versions.
- Always match PyTorch CUDA version with your system CUDA.

Missing info files:
- If the loader falls back to ZIP mode unexpectedly, confirm that:
      <KITTI_ROOT>/Kitti_infos/kitti_infos_<split>.pkl
  exists.

Empty point clouds:
- KITTI .bin files must be float32 with shape (N,4).

First-time users:
- Always test evaluation before training or explainability.
- Avoid modifying dataset loaders unless necessary.


----------------------------------------------------
7. What to Do Next
----------------------------------------------------

Once this setup is complete, you can:

- Train PointRCNN / PointPillars / PV-RCNN
- Run inference on custom frames
- Apply explainability methods (LIME, GradCAM, Occlusion)
- Extend OpenPCDet for new datasets

You are now ready to use OpenPCDet.

## Datasets and References

### Datasets

- **KITTI 3D Object Detection Dataset**  
  https://www.cvlibs.net/datasets/kitti/  
  Used for model training, evaluation, and as the reference domain for OpenPCDet and PointRCNN.

- **SONARCloud Dataset**  
  https://zenodo.org/records/16645568

### Libraries and Frameworks

- **OpenPCDet**  
  https://github.com/open-mmlab/OpenPCDet  
  Core 3D object detection framework used for PointRCNN and dataset handling.

- **PointRCNN**  
  https://arxiv.org/abs/1812.04244  
  Shi et al., *PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud*, CVPR 2019.

- **PyTorch**  
  https://pytorch.org/  
  Deep learning framework used for model definition and inference.

- **Open3D**  
  https://www.open3d.org/  
  Used for point cloud visualization and PLY export.

### Explainability Methods

- **Grad-CAM**  
  Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.

- **LIME**  
  Ribeiro et al., *Why Should I Trust You?*, KDD 2016.

- **Occlusion / Perturbation-based Explainability**  
  General perturbation-based explanation methodology adapted for 3D point clouds.

