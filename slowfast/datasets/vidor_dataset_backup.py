#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import json
import os

from . import vidor_helper as vidor_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)

import cv2

@DATASET_REGISTRY.register()
class Vidor(torch.utils.data.Dataset):
    """
    VidOR Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.VIDOR.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.VIDOR.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.VIDOR.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.VIDOR.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.VIDOR.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.VIDOR.TEST_FORCE_FLIP

        # settings for Baseline
        self.is_baseline = True if cfg.MODEL.ARCH == 'baseline' else False
        self.multigrid_enabled = cfg.MULTIGRID.ENABLE
        
        if self.cfg.MODEL.USE_TRAJECTORIES:
            self.trajectories_path = cfg.VIDOR.TRAIN_GT_TRAJECTORIES if split == 'train' else cfg.VIDOR.TEST_GT_TRAJECTORIES

        if self.cfg.MODEL.USE_SPA_CONF:
            self.human_poses_path = 'human_poses'
            # self.heatmap_size = cfg.MODEL.SPA_CONF_HEATMAP_SIZE
            self.skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                              [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
                              [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        elif self.cfg.MODEL.USE_HUMAN_POSES:
            self.human_poses_path = cfg.VIDOR.TRAIN_GT_HUMAN_POSES if split == 'train' else cfg.VIDOR.TEST_GT_HUMAN_POSES

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Load frame trajectories
        if self.cfg.MODEL.USE_TRAJECTORIES:
            if cfg.VIDOR.TEST_DEBUG:
                print('Loading trajectories...')
            with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.trajectories_path), 'r') as f:
                self._trajectories = json.load(f)

        # Load human pose features (theta; 72 points)
        if self.cfg.MODEL.USE_HUMAN_POSES:
            if cfg.VIDOR.TEST_DEBUG:
                print('Loading human poses...')
            import pickle
            with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.human_poses_path), 'rb') as f:
                self._human_poses = pickle.load(f)
        elif self.cfg.MODEL.USE_SPA_CONF:
            if cfg.VIDOR.TEST_DEBUG:
                print('Loading human poses for spatial configuration module...')
            self._human_poses_root = os.path.join(cfg.VIDOR.ANNOTATION_DIR, self.human_poses_path)

        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = vidor_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        self._instances = vidor_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = vidor_helper.get_keyframe_data(self._instances, mode=self._split)

        # import pdb; pdb.set_trace()

        # Calculate the number of used boxes.
        self._num_boxes_used = vidor_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

        def debug(idx):
            pass

        if cfg.VIDOR.TEST_DEBUG:
            debug(0)
            # pass

    def print_summary(self):
        logger.info("=== VidOR dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes, gt_boxes=None, min_scale=None, crop_size=None, n_imgs=0):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the (proposal) boxes for the current clip.
            gt_boxes: the ground truth boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        # boxes[:, [0, 2]] *= width
        # boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        crop_size = crop_size if self.multigrid_enabled and crop_size is not None else self._crop_size
        
        if self._split != 'train':
            assert gt_boxes is not None
            gt_boxes = cv2_transform.clip_boxes_to_image(gt_boxes, height, width)
            gt_boxes = [gt_boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale if not self.multigrid_enabled and min_scale is None else min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, crop_size, order="HWC", boxes=boxes, n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
            )
            if self.random_horizontal_flip:
                # random flip
                # if self.cfg.MODEL.USE_SPA_CONF and len(imgs[n_imgs].shape) != 3:
                #     for i in range(n_imgs, len(imgs) + 1):
                #         imgs[i] = np.expand_dims(imgs[i], axis=-1)
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes, 
                    n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
                )
            # elif self._split == "val":
            #     # Short side to test_scale. Non-local and STRG uses 256.
            #     imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            #     boxes, gt_boxes = cv2_transform.scale_boxes(
            #             self._crop_size, boxes[0], height, width, gt_boxes=gt_boxes[0]
            #     )
            #     boxes, gt_boxes = [boxes], [gt_boxes]
            #     imgs, boxes, gt_boxes = cv2_transform.spatial_shift_crop_list(
            #         self._crop_size, imgs, 1, boxes=boxes, gt_boxes=gt_boxes
            #     )

            #     if self._test_force_flip:
            #         imgs, boxes = cv2_transform.horizontal_flip_list(
            #             1, imgs, order="HWC", boxes=boxes, gt_boxes=gt_boxes
            #         )
        elif self._split == "val" or self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(crop_size, img) for img in imgs]
            boxes, gt_boxes = cv2_transform.scale_boxes(
                    crop_size, boxes[0], height, width, gt_boxes=gt_boxes[0]
            )
            boxes, gt_boxes = [boxes], [gt_boxes]

            if self._test_force_flip:
                # if self.cfg.MODEL.USE_SPA_CONF and len(imgs[n_imgs].shape) != 3:
                #     imgs[i] = np.expand_dims(imgs[i], axis=-1)
                    # imgs[n_imgs:] = [np.expand_dims(img, axis=-1) for img in imgs[n_imgs:]]
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes, gt_boxes=gt_boxes,
                    n_imgs=n_imgs, USE_SPA_CONF=self.cfg.MODEL.USE_SPA_CONF
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        if self.cfg.MODEL.USE_SPA_CONF:
            if len(imgs[n_imgs].shape) == 2:
                imgs[n_imgs:] = [np.expand_dims(img, axis=-1) for img in imgs[n_imgs:]]
            elif len(imgs[n_imgs].shape) > 3:
                imgs[n_imgs:] = [np.expand_dims(img.squeeze(), axis=-1) for img in imgs[n_imgs:]]
                
            # for i in range(n_imgs, len(imgs) + 1):
            #     imgs[i] = np.expand_dims(imgs[i], axis=-1)
        # try:
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
        # except:
        #     print('imgs[n_imgs].shape:', imgs[n_imgs].shape)
        #     print('len(imgs):', len(imgs))
        #     print('n_imgs:', n_imgs)
        #     import pdb; pdb.set_trace()

        # Image [0, 255] -> [0, 1].
        if self.cfg.MODEL.USE_SPA_CONF:
            imgs[:n_imgs] = [img / 255.0 for img in imgs[:n_imgs]]
        else:    
            imgs = [img / 255.0 for img in imgs]

        if self.cfg.MODEL.USE_SPA_CONF:
            imgs[:n_imgs] = [
                np.ascontiguousarray(
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs[:n_imgs]
            ]
            imgs[n_imgs:] = [
                np.ascontiguousarray(
                    img.reshape((1, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs[n_imgs:]
            ]
        else:
            imgs = [
                np.ascontiguousarray(
                    # img.reshape((3, self._crop_size, self._crop_size))
                    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
                ).astype(np.float32)
                for img in imgs
            ]

        # Do color augmentation (after divided by 255.0).
        if self.cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = imgs[n_imgs:]
            imgs = imgs[:n_imgs]
        if self._split == "train" and self._use_color_augmentation: # False
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        
        if self.cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = np.concatenate(
                [np.expand_dims(img, axis=1) for img in skeleton_imgs], axis=1
            )
            skeleton_imgs = np.ascontiguousarray(skeleton_imgs)
            skeleton_imgs = torch.from_numpy(skeleton_imgs)

        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        if gt_boxes is not None:
            gt_boxes = cv2_transform.clip_boxes_to_image(
                gt_boxes[0], imgs[0].shape[1], imgs[0].shape[2]
            )
        if self.cfg.MODEL.USE_SPA_CONF:
            return (imgs, skeleton_imgs, boxes) if gt_boxes is None else (imgs, skeleton_imgs, boxes, gt_boxes)
        else:
            return (imgs, boxes) if gt_boxes is None else (imgs, boxes, gt_boxes)

    def _images_and_boxes_preprocessing(self, imgs, boxes, gt_boxes=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        # boxes[:, [0, 2]] *= width
        # boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def draw_human_skeleton(self, human_pose, orig_width, orig_height):
        human_pose = np.array(human_pose)
        ret = np.zeros((orig_height, orig_width))
        cur_kps = np.zeros((17, 2), dtype=np.int)
        cur_kps[:, 0] = (human_pose[:, 0]).astype(np.int)
        cur_kps[:, 1] = (human_pose[:, 1]).astype(np.int)

        for j, sk in enumerate(self.skeletons):
            sk0 = sk[0] - 1
            sk1 = sk[1] - 1
            ret = cv2.line(ret, tuple(cur_kps[sk0]), tuple(cur_kps[sk1]), 0.05 * (j + 1), 1)
        
        return ret

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        if self.multigrid_enabled:
            short_cycle_idx = None
            # When short cycle is used, input index is a tupple.
            if isinstance(idx, tuple):
                idx, short_cycle_idx = idx
            
            if self._split == 'train':
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]

                if short_cycle_idx in [0, 1]:
                    crop_size = int(
                        round(
                            self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                            * self.cfg.MULTIGRID.DEFAULT_S
                        )
                    )
                
                if self.cfg.MULTIGRID.DEFAULT_S > 0:
                    # Decreasing the scale is equivalent to using a larger "span"
                    # in a sampling grid.
                    min_scale = int(
                        round(
                            float(min_scale)
                            * crop_size
                            / self.cfg.MULTIGRID.DEFAULT_S
                        )
                    )
                
                self._sample_rate = utils.get_random_sampling_rate(
                    self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                    self.cfg.DATA.SAMPLING_RATE,
                )
                self._seq_len = self._video_length * self._sample_rate
            elif self._split in ['val', 'test']:
                min_scale = crop_size = None
            else:
                raise NotImplementedError(
                    "Does not support {} mode".format(self.mode)
                )
        else:
            min_scale = crop_size = None

        video_idx, sec_idx, sec, center_idx, orig_video_idx = self._keyframe_indices[idx]
        assert orig_video_idx == '/'.join(self._image_paths[video_idx][0].split('.')[0].split('/')[-3:-1])
        
        # Get the frame idxs for current clip.
        if self.is_baseline:
            num_frames = len(self._image_paths[video_idx])
            if center_idx < 0:
                seq = [0]
            elif center_idx >= num_frames:
                seq = [num_frames - 1]
            else:
                seq = [center_idx]
        else:
            seq = utils.get_sequence(
                center_idx,
                self._seq_len // 2,
                self._sample_rate,
                num_frames=len(self._image_paths[video_idx]),
            )

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = np.array(clip_label_list[0])
        obj_classes = np.array(clip_label_list[1])
        action_labels = np.array(clip_label_list[2])
        gt_ids_to_idxs = clip_label_list[3]
        gt_idxs_to_ids = clip_label_list[4]
        ori_boxes = boxes.copy()

        if self._split != 'train':
            proposal_boxes = np.array(clip_label_list[5])
            proposal_classes = np.array(clip_label_list[6])
            proposal_scores = np.array(clip_label_list[7])
            # proposal_ids_to_idxs = clip_label_list[8]
            # proposal_idxs_to_ids = clip_label_list[9]
            
            gt_boxes = boxes
            ori_boxes = proposal_boxes.copy()
            boxes = proposal_boxes
        else:
            gt_boxes = None
        # Score is not used.
        # boxes = boxes[:, :4].copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.VIDOR.IMG_PROC_BACKEND
        )

        orig_img_width, orig_img_height = imgs[0].shape[1], imgs[0].shape[0]
        n_boxes = boxes.shape[0]
        n_imgs = len(imgs)

        if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV or self.cfg.MODEL.USE_SPA_CONF:
            assert self.cfg.MODEL.USE_TRAJECTORIES
            all_trajectories = [self._trajectories[orig_video_idx][frame] for frame in seq]
            boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)]
            trajectories = []
            for j, frame in enumerate(seq):
                trajectory = []
                all_trajectory = all_trajectories[j]
                for i in boxes_ids:
                    found = False
                    for traj in all_trajectory:
                        if traj['tid'] == i:
                            trajectory.append(list(traj['bbox'].values()))
                            found = True
                            break
                    if not found:
                        trajectory.append([0, 0, imgs[0].shape[1], imgs[0].shape[0]]) # if that object doesn't exist then use whole-img bbox
                trajectories.append(trajectory)
            # (Pdb) np.array(trajectories).shape
            # (32, 2, 4) -> 2 means n_obj
            # if self.cfg.VIDOR.TEST_DEBUG:
            #     import pdb; pdb.set_trace()
            trajectories = np.array(trajectories, dtype=np.float64)
            # trajectories = np.transpose(trajectories, [1, 0, 2])
            trajectories = trajectories.reshape(-1, 4)
            # import pdb; pdb.set_trace()
            boxes = np.concatenate((boxes, trajectories))

        if self.cfg.MODEL.USE_SPA_CONF:
            json_path = os.path.join(self._human_poses_root, orig_video_idx + '.json')
            with open(json_path, 'r') as f:
                human_poses = json.load(f)

            # human_poses = self._human_poses[orig_video_idx]
            boxes_ids = [gt_idxs_to_ids[i] for i in range(n_boxes)]
            human_poses = [human_poses[str(boxes_ids[jdx])] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]
            full_human_pose_maps = np.zeros((len(human_poses), len(seq), orig_img_height, orig_img_width))
            # scale_x = self.heatmap_size / orig_img_width
            # scale_y = self.heatmap_size / orig_img_height
            for i, human_pose in enumerate(human_poses):
                for j, frame_idx in enumerate(seq):
                    if str(frame_idx) in human_pose:
                        full_human_pose_maps[i][j] = self.draw_human_skeleton(human_pose[str(frame_idx)], orig_img_width, orig_img_height)

            # full_human_pose_maps = shape (n_person, 32, orig_img_height, orig_img_width)
            n_person = full_human_pose_maps.shape[0]
            full_human_pose_maps = full_human_pose_maps.reshape(-1, orig_img_height, orig_img_width)
            # import pdb; pdb.set_trace()
            imgs.extend([np.expand_dims(full_human_pose_maps[i], axis=-1) for i in range(n_person * self.cfg.DATA.NUM_FRAMES)])
            
            # box_maps = np.zeros((n_boxes, orig_img_height, orig_img_width))
            # n_person = human_poses.shape[0]
            # pose_maps = np.zeros((n_person, orig_img_height, orig_img_width))

        if self.cfg.VIDOR.IMG_PROC_BACKEND == "pytorch": # False
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            if gt_boxes is None:
                imgs, boxes = self._images_and_boxes_preprocessing(
                    imgs, boxes=boxes
                )
            else:
                ### NOT IMPLEMENTED! ###
                imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing(
                    imgs, boxes=boxes, gt_boxes=gt_boxes
                )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            if gt_boxes is None:
                if self.cfg.MODEL.USE_SPA_CONF:
                    imgs, skeleton_imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                        imgs, boxes=boxes, min_scale=min_scale, crop_size=crop_size,
                        n_imgs=n_imgs
                    )
                else:
                    imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                        imgs, boxes=boxes, min_scale=min_scale, crop_size=crop_size,
                        n_imgs=n_imgs
                    )
            else:
                if self.cfg.MODEL.USE_SPA_CONF:
                    imgs, skeleton_imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing_cv2(
                        imgs, boxes=boxes, gt_boxes=gt_boxes, n_imgs=n_imgs
                    )
                else:
                    imgs, boxes, gt_boxes = self._images_and_boxes_preprocessing_cv2(
                        imgs, boxes=boxes, gt_boxes=gt_boxes, n_imgs=n_imgs
                    )

        if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV or self.cfg.MODEL.USE_SPA_CONF:
            # import pdb; pdb.set_trace()
            trajectory_boxes = boxes[n_boxes:].reshape(n_boxes, -1)
            boxes = boxes[:n_boxes]

        if self.cfg.MODEL.USE_SPA_CONF:
            # import pdb; pdb.set_trace()
            # trajectory_boxes.shape = (2, 128)
            # skeleton_imgs.shape = torch.Size([1, 32, 224, 224])

            trajectory_box_masks = np.zeros((trajectory_boxes.shape[0], self.cfg.DATA.NUM_FRAMES, skeleton_imgs.shape[2], skeleton_imgs.shape[3])) # shape (n_boxes, 32, 224, 224)
            for box_id in range(trajectory_boxes.shape[0]):
                trajectory_box = trajectory_boxes[box_id].reshape(-1, 4) # shape (32, 4)
                for frame_id in range(trajectory_box.shape[0]): 
                    x1, y1, x2, y2 = int(round(trajectory_box[frame_id][0])), int(round(trajectory_box[frame_id][1])), int(round(trajectory_box[frame_id][2])), int(round(trajectory_box[frame_id][3]))
                    trajectory_box_masks[box_id][frame_id][y1:y2, x1:x2] = 1

            # import pdb; pdb.set_trace()
        
        if self.cfg.MODEL.USE_TRAJECTORIES:
            all_trajectories = [self._trajectories[orig_video_idx][frame] for frame in seq]
            boxes_ids = [gt_idxs_to_ids[i] for i in range(len(boxes))]
            trajectories = []
            for j, frame in enumerate(seq):
                trajectory = []
                all_trajectory = all_trajectories[j]
                for i in boxes_ids:
                    found = False
                    for traj in all_trajectory:
                        if traj['tid'] == i:
                            trajectory.append(list(traj['bbox'].values()))
                            found = True
                            break
                    if not found:
                        trajectory.append([0, 0, imgs[0].shape[1], imgs[0].shape[0]]) # if that object doesn't exist then use whole-img bbox
                trajectories.append(trajectory)
            # (Pdb) np.array(trajectories).shape
            # (32, 2, 4)
            # if self.cfg.VIDOR.TEST_DEBUG:
            #     import pdb; pdb.set_trace()
            trajectories = np.array(trajectories, dtype=np.float64)
            trajectories = np.transpose(trajectories, [1, 0, 2])

            width_ratio, height_ratio = imgs[0].shape[1] / orig_img_width, imgs[0].shape[0] / orig_img_height
            trajectories[:, :, 0] *= width_ratio
            trajectories[:, :, 1] *= height_ratio
            trajectories[:, :, 2] *= width_ratio
            trajectories[:, :, 3] *= height_ratio
            
            trajectories = trajectories.reshape(boxes.shape[0], -1)
            # trajectories.shape = (n_trajectories, 32*4)
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [[video_idx, sec]] * len(boxes)

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
            "obj_classes": obj_classes,
            "action_labels": action_labels,
        }    

        if self.cfg.MODEL.USE_TRAJECTORIES:
            extra_data["trajectories"] = trajectories

        if self.cfg.DETECTION.ENABLE_TOI_POOLING or self.cfg.MODEL.USE_TRAJECTORY_CONV:
            extra_data["trajectory_boxes"] = trajectory_boxes
        
        if self.cfg.MODEL.USE_SPA_CONF:
            # if self.cfg.VIDOR.TEST_DEBUG:
            #     import pdb; pdb.set_trace()
            skeleton_imgs = skeleton_imgs.reshape(n_person, self.cfg.DATA.NUM_FRAMES, skeleton_imgs.shape[2], skeleton_imgs.shape[3])
            extra_data["skeleton_imgs"] = torch.nn.functional.interpolate(skeleton_imgs, 32)

            trajectory_box_masks = torch.tensor(trajectory_box_masks)
            extra_data["trajectory_box_masks"] = torch.nn.functional.interpolate(trajectory_box_masks, 32)
        elif self.cfg.MODEL.USE_HUMAN_POSES:
            human_poses = self._human_poses[orig_video_idx]
            human_poses = np.concatenate(([[human_poses[boxes_ids[jdx]]] for jdx, obj_class in enumerate(obj_classes) if obj_class == 0]))
            human_poses = human_poses[:, seq, :]

            human_poses = human_poses.reshape(human_poses.shape[0], -1)
            extra_data["human_poses"] = human_poses
        
        if gt_boxes is not None:
            extra_data['gt_boxes'] = gt_boxes
            extra_data['proposal_classes'] = proposal_classes
            extra_data['proposal_scores'] = proposal_scores

        if self.cfg.DEMO.ENABLE:
            extra_data['orig_video_idx'] = orig_video_idx

        # print('imgs[0].shape:', imgs[0].shape, 'extra_data["boxes"][0].shape:', extra_data['boxes'][0].shape)

        return imgs, extra_data
