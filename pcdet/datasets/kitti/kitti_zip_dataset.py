import copy
import pickle
import zipfile
import io
import numpy as np
from pathlib import Path
from skimage import io as skio
import torch
import concurrent.futures as futures
import os
import tempfile
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import FarthestPointSampling
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

class KittiZipDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, save_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger
        )
        self.save_path = Path(save_path) if save_path else Path(dataset_cfg.get('INFO_DIR', '.'))
        self.info_dir = Path(getattr(dataset_cfg, 'INFO_DIR', self.save_path))

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.training_mode = 'train' if training else 'test'
        self.zip_root = Path(dataset_cfg.DATA_PATH)  # <-- zip root (read-only)
        self.small_point_cloud_samples = set()

        # ----------------------------------------------------------------- #
        # 1. Initialise zip files (fallback to filesystem if missing)
        # ----------------------------------------------------------------- #
        self.velo_zip = self.label_zip = self.calib_zip = self.image_zip = None
        try:
            self.velo_zip = zipfile.ZipFile(self.zip_root / "data_object_velodyne.zip")
            self.label_zip = zipfile.ZipFile(self.zip_root / "data_object_label_2.zip")
            self.calib_zip = zipfile.ZipFile(self.zip_root / "data_object_calib.zip")
            self.image_zip = zipfile.ZipFile(self.zip_root / "data_object_image_2.zip")
            if self.logger:
                self.logger.info(f"Opened zip files from {self.zip_root}")
        except FileNotFoundError as e:
            if self.logger:
                self.logger.warning(f"Zip files not found ({e}); will use filesystem")

        # ----------------------------------------------------------------- #
        # 2. Load sample-id list (split txt or infer from zip)
        # ----------------------------------------------------------------- #
        split_file = self.root_path / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.sample_id_list = [x.strip() for x in f.readlines()]
        else:
            self.sample_id_list = None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

        # ----------------------------------------------------------------- #
        # 3. Calibration pickle – **full Calibration objects**
        # ----------------------------------------------------------------- #
        self.calib_pkl_path = self.save_path / 'kitti_calib_data.pkl'
        self.calib_pkl = None
        if self.calib_pkl_path.exists():
            with open(self.calib_pkl_path, 'rb') as f:
                self.calib_pkl = pickle.load(f)
            if self.logger:
                self.logger.info(f"Loaded calibration pickle ({len(self.calib_pkl)} entries)")
        else:
            self.generate_calib_pkl(self.calib_pkl_path)

    # --------------------------------------------------------------------- #
    # 4. Load existing info files
    # --------------------------------------------------------------------- #
    def include_kitti_data(self, mode):
        if self.logger:
            self.logger.info('Loading KITTI-ZIP dataset')

        infos = []
        info_dir = Path(getattr(self.dataset_cfg, 'INFO_DIR', self.root_path))  # ← NEW

        for p in self.dataset_cfg.INFO_PATH[mode]:
            path = Path(p)
            if not path.is_absolute():
                path = info_dir / path   # ← use INFO_DIR, not root_path
            if not path.exists():
                if self.logger:
                    self.logger.warning(f"Info file not found: {path}")
                continue
            with open(path, 'rb') as f:
                infos.extend(pickle.load(f))

        self.kitti_infos.extend(infos)
        if self.logger:
            self.logger.info(f'Loaded {len(infos)} samples from info files')

        # Fallback: infer from zip if no infos
        if not self.kitti_infos and self.sample_id_list is None and self.velo_zip:
            ids = sorted([Path(f).stem for f in self.velo_zip.namelist()
                        if f.endswith('.bin') and f.startswith('training/velodyne/')])
            self.sample_id_list = ids
            if self.logger:
                self.logger.info(f'Inferred {len(ids)} samples from velodyne zip')

    # --------------------------------------------------------------------- #
    # 5. Build calibration pickle – **store full Calibration objects**
    # --------------------------------------------------------------------- #
    def generate_calib_pkl(self, out_path):
        """Generate and save calibration matrices (P2, R0, V2C) as a pickle."""
        calib_data = {}

        ids = (self.sample_id_list if self.sample_id_list is not None else
            [Path(f).stem for f in self.velo_zip.namelist()
                if f.endswith('.txt') and f.startswith('training/calib/')])

        for idx_str in ids:
            zip_name = f"training/calib/{idx_str}.txt"
            if self.calib_zip:
                try:
                    with self.calib_zip.open(zip_name) as f:
                        txt = f.read().decode('utf-8')
                    tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt')
                    tmp.write(txt)
                    tmp.close()
                    calib = calibration_kitti.Calibration(tmp.name)
                    os.unlink(tmp.name)
                except Exception:
                    if self.logger:
                        self.logger.warning(f"Calib {idx_str} missing in zip")
                    continue
            else:
                fs_path = self.root_split_path / 'calib' / f'{idx_str}.txt'
                if not fs_path.exists():
                    if self.logger:
                        self.logger.warning(f"Calib {idx_str} missing on disk")
                    continue
                calib = calibration_kitti.Calibration(str(fs_path))

            # ----- store only the three matrices -----
            calib_data[idx_str] = {
                'P2': np.array(calib.P2, dtype=np.float32),
                'R0': np.array(calib.R0, dtype=np.float32),
                'Tr_velo2cam': np.array(calib.V2C, dtype=np.float32)   # <-- V2C, not Tr_velo2cam
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(calib_data, f)
        self.calib_pkl = calib_data
        if self.logger:
            self.logger.info(f"Saved calibration pickle → {out_path} ({len(calib_data)} entries)")

    # --------------------------------------------------------------------- #
    # 6. Switch split
    # --------------------------------------------------------------------- #
    def set_split(self, split):
        self.split = split
        self.root_split_path = self.root_path / ('training' if split != 'test' else 'testing')
        split_file = self.root_path / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.sample_id_list = [x.strip() for x in f.readlines()]
        self.small_point_cloud_samples.clear()

    # --------------------------------------------------------------------- #
    # 7. LiDAR loader – raw_mode=True for info generation
    # --------------------------------------------------------------------- #
    def get_lidar(self, idx, raw_mode=False):
        idx_str = str(idx).zfill(6)
        min_thr = 3000 if raw_mode else 1000

        # Load point cloud
        points = np.zeros((0, 4), dtype=np.float32)
        zip_name = f"training/velodyne/{idx_str}.bin"
        if self.velo_zip:
            try:
                with self.velo_zip.open(zip_name) as f:
                    points = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"velo zip error {idx_str}: {e}")
        else:
            fs_path = self.root_split_path / 'velodyne' / f'{idx_str}.bin'
            if fs_path.exists():
                points = np.fromfile(str(fs_path), dtype=np.float32).reshape(-1, 4)

        if points.size == 0:
            self.small_point_cloud_samples.add(idx_str)
            return None if not raw_mode else np.zeros((0, 4), dtype=np.float32)

        # Range filter
        mask = ((points[:, 0] >= 0) & (points[:, 0] <= 70.4) &
                (points[:, 1] >= -40) & (points[:, 1] <= 40) &
                (points[:, 2] >= -3) & (points[:, 2] <= 1))
        points = points[mask]

        if points.shape[0] < min_thr and not raw_mode:
            self.small_point_cloud_samples.add(idx_str)
            return None

        if raw_mode:
            return points

        # CPU-only downsample/pad
        target = self.dataset_cfg.get('NUM_POINTS', {}).get(self.training_mode, 16384)

        if points.shape[0] > target:
            # Simple CPU FPS
            fps_idx = [np.random.randint(0, points.shape[0])]
            distances = np.full(points.shape[0], np.inf)
            for _ in range(1, target):
                dist = np.linalg.norm(points[:, :3] - points[fps_idx[-1], :3], axis=1)
                distances = np.minimum(distances, dist)
                fps_idx.append(np.argmax(distances))
            points = points[fps_idx]

        if points.shape[0] < target:
            w = points[:, 3] / points[:, 3].sum() if points[:, 3].sum() > 0 else None
            pad_idx = np.random.choice(points.shape[0], target - points.shape[0], replace=True, p=w)
            points = np.concatenate([points, points[pad_idx]], axis=0)

        return points

    # --------------------------------------------------------------------- #
    # 8. Image / label helpers
    # --------------------------------------------------------------------- #
    def get_image(self, idx):
        idx_int = int(idx)
        name = f"training/image_2/{idx_int:06d}.png"
        if self.image_zip:
            try:
                with self.image_zip.open(name) as f:
                    img = skio.imread(io.BytesIO(f.read())).astype(np.float32) / 255.0
            except Exception:
                img = np.zeros((0, 0, 3), dtype=np.float32)
        else:
            path = self.root_split_path / 'image_2' / f'{idx_int:06d}.png'
            try:
                img = skio.imread(path).astype(np.float32) / 255.0
            except Exception:
                img = np.zeros((0, 0, 3), dtype=np.float32)
        return img

    def get_image_shape(self, idx):
        return np.array(self.get_image(idx).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        idx_int = int(idx)
        name = f"training/label_2/{idx_int:06d}.txt"
        if self.label_zip:
            try:
                with self.label_zip.open(name) as f:
                    objs = object3d_kitti.get_objects_from_label_str(f.read().decode('utf-8'))
            except Exception:
                objs = []
        else:
            path = self.root_split_path / 'label_2' / f'{idx_int:06d}.txt'
            try:
                objs = object3d_kitti.get_objects_from_label(path)
            except Exception:
                objs = []
        return objs

    # --------------------------------------------------------------------- #
    # 9. Calibration – **return full Calibration object**
    # --------------------------------------------------------------------- #
    def get_calib(self, idx):
        """
        Load calibration from pickle, zip or filesystem.
        Returns a real calibration_kitti.Calibration instance.
        """
        idx_str = str(idx).zfill(6)

        # ----- 1. from pickle (matrix dict) -----
        if self.calib_pkl is not None and idx_str in self.calib_pkl:
            # Calibration.__init__ accepts a dict of matrices
            return calibration_kitti.Calibration(self.calib_pkl[idx_str])

        # ----- 2. from zip -----
        calib_file = f"training/calib/{idx_str}.txt"
        if self.calib_zip:
            try:
                with self.calib_zip.open(calib_file) as f:
                    txt = f.read().decode('utf-8')
                tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt')
                tmp.write(txt)
                tmp.close()
                calib = calibration_kitti.Calibration(tmp.name)
                os.unlink(tmp.name)
                return calib
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[WARNING] calib zip {idx_str}: {e}")

        # ----- 3. from filesystem -----
        fs_path = self.root_split_path / 'calib' / f'{idx_str}.txt'
        try:
            return calibration_kitti.Calibration(str(fs_path))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[WARNING] calib disk {idx_str}: {e}")
            # dummy – never reached in normal runs
            dummy = {'P2': np.zeros((3, 4)), 'R0': np.eye(3), 'V2C': np.zeros((3, 4))}
            return calibration_kitti.Calibration(dummy)

    # --------------------------------------------------------------------- #
    # 10. FOV helper
    # --------------------------------------------------------------------- #
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        pts_img, depth = calib.rect_to_img(pts_rect)
        h, w = img_shape
        flag = ((pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) &
                (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h) &
                (depth >= 0))
        return flag

    # --------------------------------------------------------------------- #
    # 11. Info generation – always use raw points
    # --------------------------------------------------------------------- #
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        def process(sample_idx):
            info = {
                'point_cloud': {'num_features': 4, 'lidar_idx': sample_idx},
                'image': {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            }
            calib = self.get_calib(sample_idx)          # <-- real Calibration

            # ---- calibration matrices (4×4) ----
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0 = np.zeros((4, 4), dtype=calib.R0.dtype)
            R0[3, 3] = 1.0
            R0[:3, :3] = calib.R0
            V2C = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            info['calib'] = {'P2': P2, 'R0_rect': R0, 'Tr_velo_to_cam': V2C}

            if not has_label:
                return info

            objs = self.get_label(sample_idx)
            annos = {}
            annos['name'] = np.array([o.cls_type for o in objs])
            annos['truncated'] = np.array([o.truncation for o in objs])
            annos['occluded'] = np.array([o.occlusion for o in objs])
            annos['alpha'] = np.array([o.alpha for o in objs])
            annos['bbox'] = np.concatenate([o.box2d.reshape(1, 4) for o in objs], axis=0)
            annos['dimensions'] = np.array([[o.l, o.h, o.w] for o in objs])
            annos['location'] = np.concatenate([o.loc.reshape(1, 3) for o in objs], axis=0)
            annos['rotation_y'] = np.array([o.ry for o in objs])
            annos['score'] = np.array([o.score for o in objs])
            annos['difficulty'] = np.array([o.level for o in objs], dtype=np.int32)

            # filter DontCare
            keep = [i for i, n in enumerate(annos['name']) if n != 'DontCare']
            for k in annos:
                annos[k] = annos[k][keep]

            num_gt = len(annos['name'])
            annos['index'] = np.arange(num_gt, dtype=np.int32)

            # lidar boxes
            loc = annos['location']
            dims = annos['dimensions']
            rots = annos['rotation_y']
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            loc_lidar[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h,
                                            -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            annos['gt_boxes_lidar'] = gt_boxes_lidar
            info['annos'] = annos

            if count_inside_pts:
                pts = self.get_lidar(sample_idx, raw_mode=True)
                if pts is None:
                    return None
                pts_rect = calib.lidar_to_rect(pts[:, :3])
                fov = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                pts_fov = pts[fov]
                corners = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                inside = np.full(num_gt, -1, dtype=np.int32)
                for i in range(num_gt):
                    inside[i] = box_utils.in_hull(pts_fov[:, :3], corners[i]).sum()
                annos['num_points_in_gt'] = inside

            return info

        ids = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as exe:
            infos = list(exe.map(process, ids))
        infos = [i for i in infos if i is not None]

        if self.logger and self.small_point_cloud_samples:
            self.logger.warning(f"Skipped {len(self.small_point_cloud_samples)} tiny clouds")
        self.small_point_cloud_samples.clear()
        return infos

    # --------------------------------------------------------------------- #
    # 12. Ground-truth database (raw points)
    # --------------------------------------------------------------------- #
    def create_groundtruth_database(self, info_path, used_classes=None, split='train'):
        db_dir = self.save_path / 'gt_database'  # <-- use save_path
        db_dir.mkdir(parents=True, exist_ok=True)
        db_info_path = self.save_path / f'kitti_dbinfos_{split}.pkl'

        if db_info_path.exists() and any(db_dir.iterdir()):
            print(f"GT database already present → {db_info_path}")
            return

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        db = {}
        for idx, info in enumerate(infos, 1):
            print(f'gt_database sample: {idx}/{len(infos)}')
            sid = info['point_cloud']['lidar_idx']
            pts = self.get_lidar(sid, raw_mode=True)
            if pts is None:
                continue
            ann = info['annos']
            names = ann['name']
            boxes = ann['gt_boxes_lidar']
            diff = ann['difficulty']
            bbox = ann['bbox']

            in_box = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(pts[:, :3]), torch.from_numpy(boxes)).numpy()

            for i in range(len(names)):
                fn = f'{sid}_{names[i]}_{i}.bin'
                path = db_dir / fn
                obj_pts = pts[in_box[i] > 0]
                obj_pts[:, :3] -= boxes[i, :3]
                obj_pts.tofile(path)

                if used_classes is None or names[i] in used_classes:
                    rel = str(path.relative_to(db_dir.parent))
                    entry = {
                        'name': names[i], 'path': rel, 'image_idx': sid,
                        'gt_idx': i, 'box3d_lidar': boxes[i],
                        'num_points_in_gt': obj_pts.shape[0],
                        'difficulty': diff[i], 'bbox': bbox[i],
                        'score': ann['score'][i] if 'score' in ann else 0
                    }
                    db.setdefault(names[i], []).append(entry)

        for c, lst in db.items():
            print(f'DB {c}: {len(lst)}')
        with open(db_info_path, 'wb') as f:
            pickle.dump(db, f)
        print(f"GT database saved → {db_info_path}")

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / f'{frame_id}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']
                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
            'image_shape': img_shape
        }

        # Handle GT boxes
        if 'annos' in info and info['annos'] is not None and len(info['annos']['name']) > 0:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            if len(annos['name']) > 0:
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
                input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list and "bbox" in annos:
                    input_dict['gt_boxes2d'] = annos["bbox"]
            else:
                input_dict['gt_names'] = np.array([])
                input_dict['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
        else:
            input_dict['gt_names'] = np.array([])
            input_dict['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)

        # Get lidar points
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if points is None:  # Handle skipped samples
                return None  # Return None to skip this sample in the dataloader
            if points.shape[0] == 0:
                points = np.zeros((self.dataset_cfg.DATA_PROCESSOR[1]['NUM_POINTS'][self.training_mode], 4), dtype=np.float32)
            else:
                if getattr(self.dataset_cfg, 'FOV_POINTS_ONLY', False):
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]
                    if points.shape[0] == 0:
                        points = np.zeros((self.dataset_cfg.DATA_PROCESSOR[1]['NUM_POINTS'][self.training_mode], 4), dtype=np.float32)

            input_dict['points'] = points

        # Optional items
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)
        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

from torch.utils.data.dataloader import default_collate

def kitti_zip_dataset_collate(batch_list):
    """
    Custom collate function for KittiZipDataset.
    Skips samples that are None or lack valid points, ensuring non-empty batches.
    """
    batch_list = [b for b in batch_list if b is not None and 'points' in b and b['points'] is not None and b['points'].size > 0]
    if len(batch_list) == 0:
        return None  # DataLoader loop must skip this batch
    return DatasetTemplate.collate_batch(batch_list)  # Use the updated collate_batch from dataset.py

# ------------------------------------------------------------------------- #
# 15. Info-generation entry point – **checks existing files**
# ------------------------------------------------------------------------- #
def create_kitti_zip_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    data_path = Path(data_path)
    save_path = Path(save_path)
    info_dir = Path(dataset_cfg.INFO_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    dataset = KittiZipDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        save_path=info_dir,           # <-- now allowed
        training=False,
        logger=common_utils.create_logger()
    )

    save_path.mkdir(parents=True, exist_ok=True)

    splits = [('train', True, True), ('val', True, True), ('test', False, False)]
    files = {
        'train': save_path / 'kitti_infos_train.pkl',
        'val':   save_path / 'kitti_infos_val.pkl',
        'trainval': save_path / 'kitti_infos_trainval.pkl',
        'test':  save_path / 'kitti_infos_test.pkl'
    }

    # ---------- 1. do we already have everything? ----------
    if all(p.exists() for p in files.values()):
        print("All info files already exist – skipping generation.")
        return

    # ---------- 2. generate missing splits ----------
    print('--------------- Generating KITTI infos ---------------')
    train_infos = val_infos = test_infos = None

    for split, label, count_pts in splits:
        dataset.set_split(split)
        infos = dataset.get_infos(num_workers=workers,
                                  has_label=label,
                                  count_inside_pts=count_pts)
        out = files[split]
        with open(out, 'wb') as f:
            pickle.dump(infos, f)
        print(f'{split} infos → {out} ({len(infos)} samples)')

        if split == 'train':
            train_infos = infos
        elif split == 'val':
            val_infos = infos
        elif split == 'test':
            test_infos = infos

    # trainval file
    with open(files['trainval'], 'wb') as f:
        pickle.dump(train_infos + val_infos, f)
    print(f'trainval infos → {files["trainval"]}')

    # ---------- 3. GT database (only for train) ----------
    if not (save_path / 'kitti_dbinfos_train.pkl').exists():
        print('--------------- Creating GT database ---------------')
        dataset.set_split('train')
        dataset.create_groundtruth_database(files['train'], split='train')
    else:
        print('GT database already present – skipping.')

    print('--------------- Data preparation finished ---------------')

# ------------------------------------------------------------------------- #
# 16. Script entry point
# ------------------------------------------------------------------------- #
if __name__ == '__main__':
    import sys, yaml
    from pathlib import Path
    from easydict import EasyDict

    for mod in list(sys.modules):
        if mod.startswith('pcdet'):
            del sys.modules[mod]

    if len(sys.argv) > 1 and sys.argv[1] == 'create_kitti_zip_infos':
        cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        create_kitti_zip_infos(
            dataset_cfg=cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=cfg.DATA_PATH,
            save_path='/netscratch/rsonawane/ex_ai/OpenPCDet/KITTI',
            workers=4
        )