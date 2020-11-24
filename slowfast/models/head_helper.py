#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union
from detectron2.layers import ROIAlign
from detectron2.layers import Conv2d, Linear, ShapeSpec
from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, Instances
from .fast_rcnn import HoiOutputLayers
from .box_head import build_hoi_head

class HOIHead(nn.Module):
    def __init__(self, cfg, resolution, scale_factor, aligned):
        # self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        super(HOIHead, self).__init__()

        # import pdb; pdb.set_trace()

        self.hoi_on = cfg.MODEL.HOI_ON
        if not self.hoi_on:
            return
        # fmt: off
        pooler_resolution      = cfg.MODEL.HOI_BOX_HEAD.POOLER_RESOLUTION
        # pooler_scales          = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio         = cfg.MODEL.HOI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.HOI_BOX_HEAD.POOLER_TYPE
        allow_person_to_person = cfg.MODEL.HOI_BOX_HEAD.ALLOW_PERSON_TO_PERSON
        # fmt: on
        self.allow_person_to_person = allow_person_to_person

        '''
        # If StandardHOROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        '''

        in_channels = cfg.MODEL.HOI_BOX_HEAD.IN_CHANNELS

        self.cfg = cfg
        
        self.hoi_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=[1.0 / scale_factor], # pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            not_box_lists=True
        )

        # self.hoi_pooler = ROIAlign(
        #     resolution,
        #     spatial_scale=1.0 / scale_factor,
        #     sampling_ratio=0,
        #     aligned=aligned,
        # )
        # self.add_module("s{}_roi".format(pathway), roi_align)

        self.hoi_head = build_hoi_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        if cfg.MODEL.SEPARATE_SUBJ_OBJ_FC:
            self.subj_head = build_hoi_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution,  width=pooler_resolution)
            )
            self.obj_head = build_hoi_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution,  width=pooler_resolution)
            )

        if cfg.DETECTION.ENABLE_TOI_POOLING:
            if cfg.MODEL.SEPARATE_SUBJ_OBJ_FC:
                self.subj_toi_head = build_hoi_head(
                    cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
                )
                self.obj_toi_head = build_hoi_head(
                    cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
                )
            else:
                self.hoi_toi_head = build_hoi_head(
                    cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
                )

        if cfg.MODEL.NORMALIZE_FEAT and cfg.MODEL.USE_BN:
            self.person_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
            self.object_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
        
        self.n_additional_feats = 0
        if cfg.MODEL.USE_FCS:
            if self.cfg.MODEL.USE_RELATIVITY_FEAT:
                self.relativity_feat_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 2 * 2 + cfg.DATA.NUM_FRAMES * 2, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                self.relativity_feat_fc2 = nn.Linear(cfg.MODEL.HOI_BOX_HEAD.FC_DIM * 2, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

                if self.cfg.MODEL.USE_BN:
                    self.relativity_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                    self.relativity_bn2 = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

            elif self.cfg.MODEL.USE_TRAJECTORIES:
                self.trajectory_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 4, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                self.trajectory_fc_person = nn.Linear(cfg.MODEL.HOI_BOX_HEAD.FC_DIM * 2, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                self.trajectory_fc_object = nn.Linear(cfg.MODEL.HOI_BOX_HEAD.FC_DIM * 2, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

                if self.cfg.MODEL.USE_BN:
                    self.trajectory_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                    self.trajectory_bn_person = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                    self.trajectory_bn_object = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

            if cfg.MODEL.USE_HUMAN_POSES:
                self.human_pose_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 72, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                self.human_pose_fc2 = nn.Linear(cfg.MODEL.HOI_BOX_HEAD.FC_DIM * 2, cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

                if self.cfg.MODEL.USE_BN:
                    self.human_pose_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)
                    self.human_pose_bn2 = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.FC_DIM)

        elif self.cfg.MODEL.USE_FC_PROJ_DIM:
            self.n_additional_feats = 0
            
            if self.cfg.MODEL.USE_RELATIVITY_FEAT:
                self.n_additional_feats += 1
                self.relativity_feat_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 2 * 2 + cfg.DATA.NUM_FRAMES * 2, self.cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM)
                
                if self.cfg.MODEL.USE_BN:
                    self.relativity_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM)

            elif self.cfg.MODEL.USE_TRAJECTORIES:
                self.n_additional_feats += 1
                self.trajectory_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 4, self.cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM // 2)
                
                if self.cfg.MODEL.USE_BN:
                    self.trajectory_bn = nn.BatchNorm1d(self.cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM // 2)
            
            if cfg.MODEL.USE_HUMAN_POSES:
                self.n_additional_feats += 1
                self.human_pose_fc = nn.Linear(cfg.DATA.NUM_FRAMES * 72, self.cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM)

                if self.cfg.MODEL.USE_BN:
                    self.human_pose_bn = nn.BatchNorm1d(cfg.MODEL.HOI_BOX_HEAD.PROJ_DIM)

        self.hoi_predictor = HoiOutputLayers(cfg, self.hoi_head.output_shape, n_additional_feats=self.n_additional_feats)

        if self.cfg.MODEL.USE_SPA_CONF:
            # self.spa_conf_pool1 = nn.MaxPool3d(
            #     kernel_size=[1, 4, 4], stride=[1, 4, 4], padding=[0, 1, 1]
            # )
            self.spa_conf_conv = nn.Conv3d(3, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
            self.spa_conf_bn = nn.BatchNorm3d(num_features=64, eps=1e-5, momentum=0.1)
            self.spa_conf_relu = nn.ReLU(self.cfg.RESNET.INPLACE_RELU)
            self.spa_conf_pool = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )
            
            self.spa_conf_conv2 = nn.Conv3d(64, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            # self.spa_conf_conv2 = nn.Conv3d(64, 256, (5, 1, 1), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
            self.spa_conf_bn2 = nn.BatchNorm3d(num_features=256, eps=1e-5, momentum=0.1)
            # self.spa_conf_pool2 = nn.MaxPool3d(
            #     kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            # )

            self.spa_conf_temporal_pool = nn.AvgPool3d([32, 1, 1], stride=1)
            self.spa_conf_fc = nn.Linear(256*8*8, self.cfg.MODEL.SPA_CONF_FC_DIM)

    @torch.no_grad()
    def construct_hopairs(self, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes=None, gt_obj_classes_lengths=None, trajectories=None, human_poses=None, skeleton_imgs=None, trajectory_box_masks=None):
        """
        Prepare person-object pairs to be used to train HOI heads.
        At training, it returns union regions of person-object proposals and assigns
            training labels. It returns ``self.hoi_batch_size_per_image`` random samples
            from pesron-object pairs, with a fraction of positives that is no larger than
            ``self.hoi_positive_sample_fraction``.
        At inference, it returns union regions of predicted person boxes and object boxes.

        Args:
            instances (list[Instances]):
                At training, proposals_with_gt. See ``self.label_and_sample_proposals``
                At inference, predicted box instances. See ``self._forward_box``

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the human-object pairs.
                Each `Instances` has the following fields:

                - union_boxes: the union region of person boxes and object boxes
                - person_boxes: person boxes in a matched sequences with union_boxes
                - object_boxes: object boxes in a matched sequences with union_boxes
                - gt_actions: the ground-truth actions that the pair is assigned.
                    Used for training HOI head.
                - person_box_scores: person box scores from box instances. Used at inference.
                - object_box_scores: object box scores from box instances. Used at inference.
                - object_box_classes: predicted box classes from box instances. Used at inference.
        """
        
        hopairs = []
        if gt_obj_classes_lengths is not None:
            end_idx_gt = 0
            gt_obj_classes_lengths = gt_obj_classes_lengths.tolist()
        end_idx = 0
        end_idx_action_labels = 0
        obj_classes_lengths = obj_classes_lengths.tolist()
        if human_poses is not None or skeleton_imgs is not None:
            end_idx_human_poses = 0
        for idx, length in enumerate(obj_classes_lengths): # same as proposal_lengths
            # for gt lengths
            if gt_obj_classes is not None and gt_obj_classes_lengths is not None:
                start_idx_gt = end_idx_gt
                length_gt = gt_obj_classes_lengths[idx]
                end_idx_gt += length_gt
                obj_classes_per_image_gt = gt_obj_classes[start_idx_gt:end_idx_gt]

                start_idx_action_labels = end_idx_action_labels
                end_idx_action_labels += length_gt ** 2
                action_labels_per_image = action_labels[start_idx_action_labels:end_idx_action_labels]

                person_idxs = (obj_classes_per_image_gt[:, 1] == 0).nonzero().squeeze(1)
                object_idxs = (obj_classes_per_image_gt[:, 1] != 0).nonzero().squeeze(1)
                if self.allow_person_to_person:
                    # Allow person to person interactions. Then all boxes will be used.
                    object_idxs = torch.arange(end_idx_gt - start_idx_gt, device=object_idxs.device)
                num_pboxes, num_oboxes = person_idxs.numel(), object_idxs.numel()
                person_idxs = person_idxs[:, None].repeat(1, num_oboxes).flatten()
                object_idxs = object_idxs[None, :].repeat(num_pboxes, 1).flatten()
                action_labels_per_image = torch.cat([
                    action_labels_per_image[i * length_gt + j, 2:].view(1, -1) for i, j in zip(person_idxs, object_idxs)
                ])
                keep = (person_idxs != object_idxs).nonzero().squeeze(1)
                gt_bbox_pair_ids = [torch.as_tensor([i, j]) for i, j in zip(person_idxs, object_idxs)]
                gt_bbox_pair_ids = torch.cat([gt_bbox_pair_ids[i].view(1, -1) for i in keep.tolist()])
                action_labels_per_image = action_labels_per_image[keep]
                hopairs_per_image = {
                    'action_labels': action_labels_per_image,
                    'gt_bbox_pair_ids': gt_bbox_pair_ids
                }
                action_labels_not_included = False
            else:
                hopairs_per_image = {}
                action_labels_not_included = True

            # for proposal lengths
            start_idx = end_idx
            end_idx += length
            boxes_per_image = bboxes[start_idx:end_idx, 1:]
            if trajectories is not None:
                trajectories_per_image = trajectories[start_idx:end_idx, 1:]
            if trajectory_box_masks is not None:
                trajectory_box_masks_per_image = trajectory_box_masks[start_idx:end_idx, 1:]
            obj_classes_per_image = obj_classes[start_idx:end_idx]
            
            if human_poses is not None or skeleton_imgs is not None:
                n_person = len(torch.nonzero(obj_classes_per_image[:, 1] == 0))
                start_idx_human_poses = end_idx_human_poses
                end_idx_human_poses += n_person
                if human_poses is not None:
                    human_poses_per_image = human_poses[start_idx_human_poses:end_idx_human_poses, 1:]
                else:
                    skeletons_per_image = skeleton_imgs[start_idx_human_poses:end_idx_human_poses, 1:]

            if action_labels_not_included:
                start_idx_action_labels = end_idx_action_labels
                end_idx_action_labels += length ** 2
                action_labels_per_image = action_labels[start_idx_action_labels:end_idx_action_labels]

            person_idxs = (obj_classes_per_image[:, 1] == 0).nonzero().squeeze(1)
            object_idxs = (obj_classes_per_image[:, 1] != 0).nonzero().squeeze(1)

            if self.allow_person_to_person:
                # Allow person to person interactions. Then all boxes will be used.
                object_idxs = torch.arange(end_idx - start_idx, device=object_idxs.device)

            num_pboxes, num_oboxes = person_idxs.numel(), object_idxs.numel()
            
            union_boxes = _pairwise_union_regions(boxes_per_image[person_idxs], boxes_per_image[object_idxs])
            # Indexing person/object boxes in a matched order.
            person_idxs = person_idxs[:, None].repeat(1, num_oboxes).flatten()
            object_idxs = object_idxs[None, :].repeat(num_pboxes, 1).flatten()
            
            # Remove self-to-self interaction.
            keep = (person_idxs != object_idxs).nonzero().squeeze(1)
            
            if action_labels_not_included:
                action_labels_per_image = torch.cat([
                    action_labels_per_image[i * length + j, 2:].view(1, -1) for i, j in zip(person_idxs, object_idxs)
                ])
                hopairs_per_image['action_labels'] = action_labels_per_image[keep]

            if len(keep) != 0:
                bbox_pair_ids = [torch.as_tensor([i, j]) for i, j in zip(person_idxs, object_idxs)]
                bbox_pair_ids = torch.cat([bbox_pair_ids[i].view(1, -1) for i in keep.tolist()])
            else:
                bbox_pair_ids = torch.tensor([])
            union_boxes = union_boxes[keep]
            person_idxs = person_idxs[keep]
            object_idxs = object_idxs[keep]

            hopairs_per_image['union_boxes'] = union_boxes
            hopairs_per_image['person_boxes'] = boxes_per_image[person_idxs]
            hopairs_per_image['object_boxes'] = boxes_per_image[object_idxs]
            hopairs_per_image['bbox_pair_ids'] = bbox_pair_ids

            if trajectories is not None:
                hopairs_per_image['person_trajectories'] = trajectories_per_image[person_idxs]
                hopairs_per_image['object_trajectories'] = trajectories_per_image[object_idxs]

                if self.cfg.MODEL.USE_RELATIVITY_FEAT:
                    # compute Relativity Feature: (c, s, m)
                    c, s, m = [], [], []
                    n_pairs = len(person_idxs)
                    n_frames = self.cfg.DATA.NUM_FRAMES # 32
                    relativity_feats = []
                    for i in range(n_pairs):
                        s = hopairs_per_image['person_trajectories'][i].view(n_frames, 4)
                        o = hopairs_per_image['object_trajectories'][i].view(n_frames, 4)
                        
                        c_s = torch.cat([(s[:, 0] + s[:, 2] / 2).view(-1, 1), (s[:, 1] + s[:, 3] / 2).view(-1, 1)], dim=1)
                        c_o = torch.cat([(o[:, 0] + o[:, 2] / 2).view(-1, 1), (o[:, 1] + o[:, 3] / 2).view(-1, 1)], dim=1)
                        delta_c = c_s - c_o # torch.Size ([32, 2])

                        s_s = s[:, 2:4]
                        s_o = o[:, 2:4]
                        delta_s = s_s - s_o # torch.Size ([32, 2])

                        delta_m = torch.cat([(delta_c[i] - delta_c[i - 1]).view(-1, 1) for i in range(1, delta_c.shape[0])], dim=1) # torch.Size ([30, 2])
                        
                        relativity_feat = torch.cat([delta_c.view(-1), delta_s.view(-1), delta_m.view(-1)]).view(1, -1)
                        relativity_feats.append(relativity_feat)
                    hopairs_per_image['relativity_feats'] = torch.cat(relativity_feats)

                if self.cfg.MODEL.USE_HUMAN_POSES:
                    assert len(torch.unique(person_idxs)) == n_person
                    n_repeat = len(person_idxs) // n_person
                    hopairs_per_image['human_poses'] = torch.cat([
                        human_poses_per_image[i].repeat(n_repeat, 1) for i in range(n_person)
                    ])
                elif self.cfg.MODEL.USE_SPA_CONF:
                    # (Pdb) skeleton_imgs.shape
                    # torch.Size([40, 33, 224, 224])
                    # (Pdb) trajectory_box_masks.shape
                    # torch.Size([56, 33, 224, 224])
                    try:
                        assert len(torch.unique(person_idxs)) == n_person
                    except:
                        import pdb; pdb.set_trace()
                    n_repeat = len(person_idxs) // n_person
                    skeletons = torch.cat([
                        skeletons_per_image[i].repeat(n_repeat, 1, 1, 1) for i in range(n_person)
                    ])
                    trajectory_box_masks_person = trajectory_box_masks_per_image[person_idxs]
                    trajectory_box_masks_object = trajectory_box_masks_per_image[object_idxs]

                    assert skeletons.shape == trajectory_box_masks_object.shape == trajectory_box_masks_person.shape # sanity check
                    # import pdb; pdb.set_trace()
                    hopairs_per_image['spa_conf_maps'] = torch.cat([
                        torch.cat([
                            torch.cat([
                                skeletons[i, j].unsqueeze(0), trajectory_box_masks_person[i, j].unsqueeze(0), trajectory_box_masks_object[i, j].unsqueeze(0)
                            ]).unsqueeze(0) for j in range(skeletons.shape[1])
                        ]).unsqueeze(0) for i in range(skeletons.shape[0])
                    ]).permute(0, 2, 1, 3, 4)

            # if self.training:
            hopairs_per_image['person_idxs'] = person_idxs
            hopairs_per_image['object_idxs'] = object_idxs

            hopairs.append(hopairs_per_image)

        return hopairs
    
    def forward(self, features, bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes=None, gt_obj_classes_lengths=None, trajectories=None, human_poses=None, toi_pooled_features=None, trajectory_boxes=None, skeleton_imgs=None, trajectory_box_masks=None):
        # (Pdb) features.shape
        # torch.Size([16, 2304, 14, 14])
        # (Pdb) bboxes.shape
        # torch.Size([59, 5])
        # (Pdb) obj_classes.shape
        # torch.Size([59, 2])
        # (Pdb) bboxes[:5]
        # tensor([[  0.0000, 223.0000, 135.6844, 223.0000, 157.5156],
        #         [  0.0000, 206.4203, 141.9219, 222.0141, 159.0750],
        #         [  0.0000,   0.0000, 130.2266,  73.8734, 211.3141],
        #         [  1.0000,   0.0000,  85.0000,   0.0000, 223.0000],
        #         [  1.0000,   0.0000, 183.7000,  67.7500, 223.0000]], device='cuda:0')
        # (Pdb) obj_classes[:5]
        # tensor([[0., 0.],
        #         [0., 0.],
        #         [0., 0.],
        #         [1., 0.],
        #         [1., 0.]], device='cuda:0')

        hopairs = self.construct_hopairs(bboxes, obj_classes, obj_classes_lengths, action_labels, gt_obj_classes, gt_obj_classes_lengths, trajectories, human_poses, skeleton_imgs, trajectory_box_masks)
        # NOTE that requires_grad for data in `hopairs` are False!!! Set to True before proceeding!
        
        spa_conf_maps = None
        if self.cfg.MODEL.USE_SPA_CONF:
            spa_conf_maps = torch.cat([x['spa_conf_maps'] for x in hopairs]) # torch.Size([[53, 3, 32, 224, 224]])
            try:
                # spa_conf_maps = F.interpolate(spa_conf_maps, (spa_conf_maps.shape[2], 32, 32))
                assert spa_conf_maps.shape[3] == spa_conf_maps.shape[4] == 32
            except:
                import pdb; pdb.set_trace()
            
            spa_conf_maps = self.spa_conf_conv(spa_conf_maps) 
            spa_conf_maps = self.spa_conf_bn(spa_conf_maps)
            spa_conf_maps = self.spa_conf_relu(spa_conf_maps)
            spa_conf_maps = self.spa_conf_pool(spa_conf_maps) 

            spa_conf_maps = self.spa_conf_conv2(spa_conf_maps) 
            spa_conf_maps = self.spa_conf_bn2(spa_conf_maps)
            spa_conf_maps = self.spa_conf_relu(spa_conf_maps)

            spa_conf_maps = self.spa_conf_temporal_pool(spa_conf_maps)
            
            if self.cfg.VIDOR.TEST_DEBUG:
                print('spa_conf_maps.shape:', spa_conf_maps.shape)
            spa_conf_maps = torch.flatten(spa_conf_maps, start_dim=1)
            try:
                spa_conf_maps = self.spa_conf_fc(spa_conf_maps)
                spa_conf_maps = self.spa_conf_relu(spa_conf_maps)
            except:
                import pdb; pdb.set_trace()

        # The pipeline for ToI pooled features
        if self.cfg.DETECTION.ENABLE_TOI_POOLING:
            # toi_pooled_features = torch.Size([nb_bboxes(e.g. 54), 2304(2048+256), 7, 7])
            start_idx = end_idx = 0
            person_features = []
            object_features = []
            for i in range(len(obj_classes_lengths)):
                end_idx += obj_classes_lengths[i].item()

                features_per_img = toi_pooled_features[start_idx:end_idx]
                person_feature = features_per_img[hopairs[i]['person_idxs']]
                object_feature = features_per_img[hopairs[i]['object_idxs']]

                person_features.append(person_feature)
                object_features.append(object_feature)

                start_idx = end_idx
            person_features = torch.cat(person_features)
            object_features = torch.cat(object_features)
        else:
            person_features = self.hoi_pooler([features], [x['person_boxes'] for x in hopairs])
            object_features = self.hoi_pooler([features], [x['object_boxes'] for x in hopairs])
        # features = [features[f] for f in self.in_features]
        union_features  = self.hoi_pooler([features], [x['union_boxes'] for x in hopairs])
        # (Pdb) union_features.shape
        # torch.Size([96, 2304, 7, 7])

        # if self.cfg.VIDOR.TEST_DEBUG:
        #     import pdb; pdb.set_trace()

        union_features = self.hoi_head(union_features) # torch.Size([96, 512])
        if self.cfg.DETECTION.ENABLE_TOI_POOLING:
            if self.cfg.MODEL.SEPARATE_SUBJ_OBJ_FC:
                person_features = self.subj_toi_head(person_features)
                object_features = self.obj_toi_head(object_features)
            else:
                person_features = self.hoi_toi_head(person_features)
                object_features = self.hoi_toi_head(object_features)
        else:
            if self.cfg.MODEL.SEPARATE_SUBJ_OBJ_FC:
                person_features = self.subj_head(person_features)
                object_features = self.obj_head(object_features)
            else:
                person_features = self.hoi_head(person_features)
                object_features = self.hoi_head(object_features)
        
        if self.cfg.MODEL.NORMALIZE_FEAT:
            if self.cfg.MODEL.USE_BN:
                person_features = self.person_bn(person_features)
                object_features = self.object_bn(object_features)
            person_features = F.relu(person_features)
            object_features = F.relu(object_features)

        if trajectories is not None:
            if self.cfg.MODEL.USE_TRAJECTORY_CONV:
                import pdb; pdb.set_trace()
            elif self.cfg.MODEL.USE_RELATIVITY_FEAT:
                relativity_feats = torch.cat([x['relativity_feats'] for x in hopairs]) # torch.Size([96, 190])
                
                if self.cfg.MODEL.USE_FCS:
                    relativity_feats = self.relativity_feat_fc(relativity_feats)
                    if self.cfg.MODEL.USE_BN:
                        relativity_feats = self.relativity_bn(relativity_feats)
                    relativity_feats = F.relu(relativity_feats)

                    union_features = self.relativity_feat_fc2(torch.cat([union_features, relativity_feats], dim=1))
                    if self.cfg.MODEL.USE_BN:
                        union_features = self.relativity_bn2(union_features)
                    union_features = F.relu(union_features)

                elif self.cfg.MODEL.USE_FC_PROJ_DIM:
                    relativity_feats = self.relativity_feat_fc(relativity_feats)
                    if self.cfg.MODEL.USE_BN:
                        relativity_feats = self.relativity_bn(relativity_feats)
                    relativity_feats = F.relu(relativity_feats)
                    union_features = torch.cat([union_features, relativity_feats], dim=1)
                else:
                    union_features = torch.cat([union_features, relativity_feats], dim=1)
            else: # USE_TRAJECTORY
                person_trajectories = torch.cat([x['person_trajectories'] for x in hopairs]) # torch.Size([96, 128])
                object_trajectories = torch.cat([x['object_trajectories'] for x in hopairs])
                
                if self.cfg.MODEL.USE_FCS:
                    assert not self.cfg.MODEL.USE_FC_PROJ_DIM

                    person_trajectories = self.trajectory_fc(person_trajectories)
                    if self.cfg.MODEL.USE_BN:
                        person_trajectories = self.trajectory_bn(person_trajectories)
                    person_trajectories = F.relu(person_trajectories)
                    
                    object_trajectories = self.trajectory_fc(object_trajectories)
                    if self.cfg.MODEL.USE_BN:
                        object_trajectories = self.trajectory_bn(object_trajectories)
                    object_trajectories = F.relu(object_trajectories)

                    person_features = self.trajectory_fc_person(torch.cat((person_features, person_trajectories), dim=1))
                    if self.cfg.MODEL.USE_BN:
                        person_features = self.trajectory_fc_person(person_features)
                    person_features = F.relu(person_features)
                    
                    object_features = self.trajectory_fc_object(torch.cat((object_features, object_trajectories), dim=1))
                    if self.cfg.MODEL.USE_BN:
                        object_features = self.trajectory_fc_object(object_features)
                    object_features = F.relu(object_features)
                elif self.cfg.MODEL.USE_FC_PROJ_DIM:
                    person_trajectories = self.trajectory_fc(person_trajectories)
                    object_trajectories = self.trajectory_fc(object_trajectories)

                    if self.cfg.MODEL.USE_BN:
                        person_trajectories = self.trajectory_bn(person_trajectories)
                        object_trajectories = self.trajectory_bn(object_trajectories)

                    person_trajectories = F.relu(person_trajectories)
                    person_trajectories = F.relu(person_trajectories)

                    person_features = torch.cat((person_features, person_trajectories), dim=1)
                    object_features = torch.cat((object_features, object_trajectories), dim=1)
                else:
                    person_features = torch.cat((person_features, person_trajectories), dim=1)
                    object_features = torch.cat((object_features, object_trajectories), dim=1)

            if self.cfg.MODEL.USE_HUMAN_POSES:
                human_poses = torch.cat([x['human_poses'] for x in hopairs]) # torch.Size([96, 2304])

                if self.cfg.MODEL.USE_FCS:
                    assert not self.cfg.MODEL.USE_FC_PROJ_DIM

                    human_poses = self.human_pose_fc(human_poses)
                    if self.cfg.MODEL.USE_BN:
                        human_poses = self.human_pose_bn(human_poses)
                    human_poses = F.relu(human_poses)

                    person_features = self.human_pose_fc2(torch.cat((person_features, human_poses), dim=1))
                    if self.cfg.MODEL.USE_BN:
                        person_features = self.human_pose_bn2(person_features)
                    person_features = F.relu(person_features)

                elif self.cfg.MODEL.USE_FC_PROJ_DIM:
                    human_poses = self.human_pose_fc(human_poses)
                    if self.cfg.MODEL.USE_BN:
                        human_poses = self.human_pose_bn(human_poses)
                    human_poses = F.relu(human_poses)

                    person_features = torch.cat((person_features, human_poses), dim=1)
                else:
                    print('USE_HUMAN_POSES with simple concat IS NOT implemented!')
                    assert False
        
        hoi_predictions = self.hoi_predictor(union_features, person_features, object_features, spa_conf_maps=spa_conf_maps)

        del union_features, person_features, object_features, features

        if gt_obj_classes is None:
            return hoi_predictions, cat([x['action_labels'] for x in hopairs], dim=0), torch.cat([x['bbox_pair_ids'] for x in hopairs])
        else:
            return hoi_predictions, cat([x['action_labels'] for x in hopairs], dim=0), torch.cat([x['bbox_pair_ids'] for x in hopairs]), torch.cat([x['gt_bbox_pair_ids'] for x in hopairs]) # , hopairs


def _pairwise_union_regions(boxes1, boxes2):
    """
    Given two lists of boxes of size N and M, compute the union regions between
    all N x M pairs of boxes. The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        unions: Boxes
    """
    
    X1 = torch.min(boxes1[:, None, 0], boxes2[:, 0]).flatten()
    Y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1]).flatten()
    X2 = torch.max(boxes1[:, None, 2], boxes2[:, 2]).flatten()
    Y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3]).flatten()

    unions = torch.stack([X1, Y1, X2, Y2], dim=1)
    # unions = Boxes(unions) # BoxMode.XYXY_ABS

    return unions


class ResNetPoolHead(nn.Module):
    """
    ResNe(X)t HoI head.
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        is_baseline=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetPoolHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.is_baseline = is_baseline

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                temporal_pool = nn.AvgPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)
            
            '''
            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)

            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)
            '''

        '''
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
        '''

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        # import pdb; pdb.set_trace()
        # (Pdb) inputs[0].shape
        # torch.Size([batch_size, 2048, 8, 14, 14])
        # (Pdb) inputs[1].shape
        # torch.Size([batch_size, 256, 32, 14, 14])
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway)) # AvgPool3d(kernel_size=[8(or 32), 1, 1])
            out = t_pool(inputs[pathway]) # torch.Size([batch_size, 2048, 8 -> 1, 14, 14]) or torch.Size([batch_size, 256, 32 -> 1, 14, 14])
            
            if not self.is_baseline:
                assert out.shape[2] == 1
                out = torch.squeeze(out, 2) # -> torch.Size([batch_size, 2048, 14, 14]) or torch.Size([batch_size, 256, 14, 14])

            '''
            roi_align = getattr(self, "s{}_roi".format(pathway)) # ROIAlign(output_size=[7,7])
            out = roi_align(out, bboxes) # torch.Size([nb_bboxes, 2048(or 256), 7, 7])

            s_pool = getattr(self, "s{}_spool".format(pathway)) # MaxPool2d(kernel_size=[7, 7])
            pool_out.append(s_pool(out)) # torch.Size([nb_bboxes, 2048(or 256), 1, 1])
            '''
            pool_out.append(out)
    
        # B C H W.
        x = torch.cat(pool_out, 1) # torch.Size([nb_bboxes, 2304(2048+256), 14, 14])
        
        '''
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x) # Linear(in_features=2304, out_features=80, bias=True)
        x = self.act(x) # Sigmoid()
        '''
        return x


class ResNetToIHead(nn.Module):
    """
    ResNe(X)t HoI head.
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        num_frames,
        alpha,
        is_baseline=False,
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetToIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.is_baseline = is_baseline
        
        # for toi pooling
        self.num_frames = num_frames
        self.alpha = alpha

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                temporal_pool = nn.AvgPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)
            
            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            # import pdb; pdb.set_trace()

            '''
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)
            '''

        '''
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
        '''

    def forward(self, inputs, bboxes, trajectory_boxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        # import pdb; pdb.set_trace()
        # (Pdb) inputs[0].shape
        # torch.Size([batch_size, 2048, 8, 14, 14])
        # (Pdb) inputs[1].shape
        # torch.Size([batch_size, 256, 32, 14, 14])

        for pathway in range(self.num_pathways):
            # (Pdb) inputs[pathway].shape
            # torch.Size([16, 2048, 8 or 32, 14, 14])
            roi_align = getattr(self, "s{}_roi".format(pathway)) # ROIAlign(output_size=[7,7])
            total_frames = trajectory_boxes[0,1:].shape[0] // 4 # 128 / 4 = 32
            n_frames = inputs[pathway].shape[2] # 8 or 32 frames
            # linspace = total_frames // n_frames # 4 or 1
            # import pdb; pdb.set_trace()
            if n_frames < self.num_frames:
                linspace = torch.linspace(0, self.num_frames - 1, self.num_frames // self.alpha).long()
            else:
                linspace = torch.tensor([i for i in range(self.num_frames)]).long()
            
            frame_pooled_feats = []
            for i in range(len(linspace)):
                frame_feat_map = inputs[pathway][:, :, i] # torch.Size([16, 2048, 14, 14])
                trajectory_boxes_frame = torch.cat([
                    trajectory_boxes[:, 0].view(-1, 1), 
                    trajectory_boxes[:, 1 + linspace[i] * 4:1 + (linspace[i] + 1) * 4]
                ], dim=1)
                
                frame_pooled_feat = roi_align(frame_feat_map, trajectory_boxes_frame)
                frame_pooled_feats.append(frame_pooled_feat.unsqueeze(2))
            frame_pooled_feats = torch.cat(frame_pooled_feats, dim=2) # torch.Size([54, 2048, 8, 7, 7]) or torch.Size([54, 256, 32, 7, 7])
            
            # inputs[pathway].shape = torch.Size([16, 2048, 8, 14, 14])
            # bboxes.shape = torch.Size([54, 5])
            # bboxes = tensor([[  0.0000,   0.0000,  10.1125,  94.2875, 219.9031],
            #                  [  0.0000, 143.9594,   0.0000, 223.0000, 223.0000],
            #                  [  1.0000, 127.4828, 158.3406, 223.0000, 223.0000],...

            # out = roi_align(out, bboxes) # torch.Size([nb_bboxes, 2048(or 256), 7, 7])

            t_pool = getattr(self, "s{}_tpool".format(pathway)) # AvgPool3d(kernel_size=[8(or 32), 1, 1])
            out = t_pool(frame_pooled_feats) # torch.Size([batch_size, 2048, 8 -> 1, 7, 7]) or torch.Size([batch_size, 256, 32 -> 1, 7, 7])
            out = torch.squeeze(out, 2) # -> torch.Size([batch_size, 2048, 7, 7]) or torch.Size([batch_size, 256, 7, 7])

            '''
            s_pool = getattr(self, "s{}_spool".format(pathway)) # MaxPool2d(kernel_size=[7, 7])
            pool_out.append(s_pool(out)) # torch.Size([nb_bboxes, 2048(or 256), 1, 1])
            '''
            pool_out.append(out)
    
        # B C H W.
        x = torch.cat(pool_out, 1) # torch.Size([nb_bboxes, 2304(2048+256), 7, 7])
        
        '''
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x) # Linear(in_features=2304, out_features=80, bias=True)
        x = self.act(x) # Sigmoid()
        '''
        return x


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)

            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        # import pdb; pdb.set_trace()
        # (Pdb) inputs[0].shape
        # torch.Size([batch_size, 2048, 8, 14, 14])
        # (Pdb) inputs[1].shape
        # torch.Size([batch_size, 256, 32, 14, 14])
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway)) # AvgPool3d(kernel_size=[8(or 32), 1, 1])
            out = t_pool(inputs[pathway]) # torch.Size([batch_size, 2048, 8 -> 1, 14, 14]) or torch.Size([batch_size, 256, 32 -> 1, 14, 14])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2) # -> torch.Size([batch_size, 2048, 14, 14]) or torch.Size([batch_size, 256, 14, 14])

            roi_align = getattr(self, "s{}_roi".format(pathway)) # ROIAlign(output_size=[7,7])
            # bboxes = tensor([[0., 0., 1., 0., 1.]], device='cuda:0')
            out = roi_align(out, bboxes) # torch.Size([nb_bboxes, 2048(or 256), 7, 7])

            s_pool = getattr(self, "s{}_spool".format(pathway)) # MaxPool2d(kernel_size=[7, 7])
            pool_out.append(s_pool(out)) # torch.Size([nb_bboxes, 2048(or 256), 1, 1])

        # B C H W.
        x = torch.cat(pool_out, 1) # torch.Size([nb_bboxes, 2304(2048+256), 1, 1])

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x) # Linear(in_features=2304, out_features=80, bias=True)
        x = self.act(x) # Sigmoid()
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x
