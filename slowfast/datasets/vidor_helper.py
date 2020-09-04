#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict
from fvcore.common.file_io import PathManager
import json
import torch

logger = logging.getLogger(__name__)

# FPS = 30
# AVA_VALID_FRAMES = range(902, 1799)


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.VIDOR.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.VIDOR.TRAIN_LISTS if is_train else cfg.VIDOR.TEST_LISTS
        )
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with PathManager.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 4
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.VIDOR.FRAME_DIR, row[3])
                )

    image_paths = [image_paths[i] for i in range(len(image_paths))]

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )

    return image_paths, video_idx_to_name

def load_hoi_instances_and_labels(cfg, mode):
    # Return a dict of:
    #   proposals
    #   gt_instances

    gt_lists = cfg.VIDOR.TRAIN_GT_BOX_LISTS if mode == "train" else cfg.VIDOR.TEST_GT_BOX_LISTS
    # pred_lists = (
    #     cfg.VIDOR.TRAIN_PREDICT_BOX_LISTS
    #     if mode == "train"
    #     else cfg.VIDOR.TEST_PREDICT_BOX_LISTS
    # )
    pred_lists = []
    ### ONLY FOR DEBUG PURPOSE: ONLY USE GT BBOXES FOR BOTH TRAIN & VAL! ###

    ann_filenames = [
        os.path.join(cfg.VIDOR.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    detect_thresh = cfg.VIDOR.DETECTION_SCORE_THRESH
    all_hoi_instances = {}
    count = 0
    # unique_box_count = 0
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with PathManager.open(filename, "r") as f:
            anns = json.load(f)
        
        for ann in anns:
            pass
    
    return None

def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    ### ONLY FOR DEBUG PURPOSE: ONLY USE GT BBOXES FOR BOTH TRAIN & VAL! ###
    # gt_lists = cfg.VIDOR.TRAIN_GT_BOX_LISTS if mode == "train" else []
    gt_lists = cfg.VIDOR.TRAIN_GT_BOX_LISTS if mode == "train" else cfg.VIDOR.TEST_GT_BOX_LISTS
    # pred_lists = (
    #     cfg.VIDOR.TRAIN_PREDICT_BOX_LISTS
    #     if mode == "train"
    #     else cfg.VIDOR.TEST_PREDICT_BOX_LISTS
    # )
    pred_lists = [] if mode == "train" else cfg.VIDOR.TEST_PREDICT_BOX_LISTS
    ### ONLY FOR DEBUG PURPOSE: ONLY USE GT BBOXES FOR BOTH TRAIN & VAL! ###

    ann_filenames = [
        os.path.join(cfg.VIDOR.ANNOTATION_DIR, filename)
        for filename in gt_lists # + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists)#  + [False] * len(pred_lists)

    detect_thresh = cfg.VIDOR.DETECTION_SCORE_THRESH
    all_boxes = {}
    count = 0
    unique_box_count = 0
    end_cur_frame_idx = 0

    if mode != 'train':
        val_instance_pred_path = os.path.join(cfg.VIDOR.ANNOTATION_DIR, pred_lists[0])
        with open(val_instance_pred_path, 'rb') as f:
            inst = torch.load(f)
        inst = {ins['image_id']:ins['instances'] for ins in inst}

    if cfg.VIDOR.TEST_WITH_GT:
        logger.info('testing with ground truth bboxes...')

    instances = {}
    proposals = []
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with PathManager.open(filename, "r") as f:
            anns = json.load(f)
        
        while end_cur_frame_idx < len(anns):
            start_cur_frame_idx = end_cur_frame_idx
            all_obj_ids = set()
            while end_cur_frame_idx < len(anns) and anns[end_cur_frame_idx]['video_id'] == anns[start_cur_frame_idx]['video_id'] and anns[end_cur_frame_idx]['frame_id'] == anns[start_cur_frame_idx]['frame_id']:
                all_obj_ids.add(anns[end_cur_frame_idx]['person_id'])
                all_obj_ids.add(anns[end_cur_frame_idx]['object_id'])
                end_cur_frame_idx += 1

            # import pdb; pdb.set_trace()
            # if mode in ['val', 'test']:
            #     if anns[start_cur_frame_idx]['video_id'] + '_' + str(f'{anns[start_cur_frame_idx]['middle_frame_timestamp']+1:06d}') in 

            cur_frame_anns = anns[start_cur_frame_idx:end_cur_frame_idx]
            ids_to_boxes = {key:value for key, value in zip(list(all_obj_ids), range(len(all_obj_ids)))}
            boxes_to_ids = {value:key for key, value in zip(list(all_obj_ids), range(len(all_obj_ids)))}

            if is_gt_box:
                gt_boxes = {} # [[] for _ in range(len(ids_to_boxes))]
                gt_classes = {} # [-1 for _ in range(len(ids_to_boxes))]
                gt_actions = {} # [[] for _ in range(len(ids_to_boxes))]
            else:
                proposal_boxes = {} # [[] for _ in range(len(ids_to_boxes))]
                is_person = {} # [-1 for _ in range(len(ids_to_boxes))]
            
            for ann in cur_frame_anns:
                if is_gt_box:
                    person_id_in_boxes = ids_to_boxes[ann['person_id']]
                    # if ann['object_id'] not in ids_to_boxes:
                    #     import pdb; pdb.set_trace()
                    object_id_in_boxes = ids_to_boxes[ann['object_id']]

                    if person_id_in_boxes not in gt_boxes:
                        gt_boxes[person_id_in_boxes] = [
                            ann['person_box']['xmin'],
                            ann['person_box']['ymin'],
                            ann['person_box']['xmax'],
                            ann['person_box']['ymax']
                        ]
                    if object_id_in_boxes not in gt_boxes:
                        gt_boxes[object_id_in_boxes] = [
                            ann['object_box']['xmin'],
                            ann['object_box']['ymin'],
                            ann['object_box']['xmax'],
                            ann['object_box']['ymax']
                        ]

                    if person_id_in_boxes not in gt_classes:
                        gt_classes[person_id_in_boxes] = 0 # Person cls id in VidOR is 0
                    if object_id_in_boxes not in gt_classes:
                        gt_classes[object_id_in_boxes] = ann['object_class']

                    if person_id_in_boxes not in gt_actions:
                        gt_actions[person_id_in_boxes] = [[0] * 50 for _ in range(len(ids_to_boxes))]
                    gt_actions[person_id_in_boxes][object_id_in_boxes][ann['action_class']] = 1
                    if object_id_in_boxes not in gt_actions:
                        gt_actions[object_id_in_boxes] = [[0] * 50 for _ in range(len(ids_to_boxes))]
                else:
                    raise NotImplementedError('TO BE IMPLEMENTED')
                    # proposal_boxes & is_person
            
            if is_gt_box:
                new_video_id = ann['video_folder'] + '/' + ann['video_id']
                if new_video_id not in instances:
                    instances[new_video_id] = {}
                instances[new_video_id][ann['middle_frame_timestamp']] = {
                    'gt_boxes': [gt_boxes[key] for key in sorted(gt_boxes.keys())],
                    'gt_classes': [gt_classes[key] for key in sorted(gt_classes.keys())],
                    'gt_actions': [gt_actions[key] for key in sorted(gt_actions.keys())],
                    'video_fps': ann['video_fps'],
                    'gt_ids_to_idxs': {
                        boxes_to_ids[key]:idx for idx, key in enumerate(sorted(gt_boxes.keys()))
                    }, # for adding continunous bboxes
                    'gt_idxs_to_ids': {
                        idx:boxes_to_ids[key] for idx, key in enumerate(sorted(gt_boxes.keys()))
                    }, # for adding continunous bboxes
                }
            else:
                assert False # not implemented section
                proposals.append({
                    'proposal_boxes': proposal_boxes.values(),
                    'is_person': is_person.values(),
                })

            if mode != 'train':
                # import pdb; pdb.set_trace()
                image_id = new_video_id + '_' + f"{ann['middle_frame_timestamp']+1:06d}"
                if image_id in inst:
                    proposal_boxes = []
                    proposal_classes = []
                    proposal_scores = []
                    
                    if cfg.VIDOR.TEST_WITH_GT: # FOR DEBUG PURPOSE!
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_boxes'] = instances[new_video_id][ann['middle_frame_timestamp']]['gt_boxes']
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_classes'] = instances[new_video_id][ann['middle_frame_timestamp']]['gt_classes']
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_scores'] = [1.0] * len(instances[new_video_id][ann['middle_frame_timestamp']]['proposal_boxes'])
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_boxes_to_ids'] = instances[new_video_id][ann['middle_frame_timestamp']]['gt_ids_to_idxs']
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_ids_to_boxes'] = instances[new_video_id][ann['middle_frame_timestamp']]['gt_idxs_to_ids']
                    else:
                        # Skip no instance predicted validation frames for now
                        if len(inst[image_id]) == 0:
                            print(f'image id {image_id} skipped due to no instance predicted!')
                            del instances[new_video_id][ann['middle_frame_timestamp']]
                            continue

                        for annot in inst[image_id]: # for each inst prediction in an image
                            if annot['score'] < detect_thresh: # 0.0
                                break
                            bb = annot['bbox']
                            x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
                            proposal_boxes.append([x1, y1, x2, y2])
                            proposal_classes.append(annot['category_id'])
                            proposal_scores.append(annot['score'])
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_boxes'] = proposal_boxes
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_classes'] = proposal_classes
                        instances[new_video_id][ann['middle_frame_timestamp']]['proposal_scores'] = proposal_scores
                        ### NOTE that unique boxes tracking mechanism not implemented yet -> currently only work for GT testing ###
                        # instances[new_video_id][ann['middle_frame_timestamp']]['proposal_boxes_to_ids'] = instances[new_video_id][ann['middle_frame_timestamp']]['boxes_to_ids']
                        ### NOTE that unique boxes tracking mechanism not implemented yet -> currently only work for GT testing ###
            
            # import pdb; pdb.set_trace()
    
    # if mode != 'train':
    #     pred_instances = {}
    #     val_instance_pred_path = os.path.join(cfg.VIDOR.ANNOTATION_DIR, pred_lists[0])
    #     with open(val_instance_pred_path, 'rb') as f:
    #         inst = torch.load(f)
    #     for cur_frame_anns in inst:
    #         video_id = cur_frame_anns['image_id'].split('_')[0]
    #         sec = int(cur_frame_anns['image_id'].split('_')[1]) - 1
    #         if video_id not in pred_instances:
    #             pred_instances[video_id] = {}
            
    #         proposal_boxes = []
    #         pred_classes = []
    #         scores = []
    #         for ann in cur_frame_anns['instances']:
    #             bb = ann['bbox']
    #             x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
    #             proposal_boxes.append([x1, y1, x2, y2])
    #             pred_classes.append(ann['category_id'])
    #             scores.append(ann['score'])
    #         pred_instances[video_id][sec] = {
    #             'proposal_boxes': proposal_boxes,
    #             'pred_classes': pred_classes,
    #             'scores': scores,
    #         }
    
    '''
    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)
    '''
    
    return instances # or proposals


def get_keyframe_data(instances, mode):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec, fps):
        """
        Convert time index (in second) to frame index.
        0: 0
        30: 1
        """
        return (sec - 0) * fps

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    pred_count = 0
    # keyframes_set = set()
    for i, video_idx in enumerate(instances.keys()):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in instances[video_idx].keys():
            # if sec not in AVA_VALID_FRAMES:
            #     continue
            if len(instances[video_idx][sec]['gt_boxes']) > 0:
                keyframe_indices.append(
                    (i, sec_idx, sec, sec_to_frame(sec, instances[video_idx][sec]['video_fps']), video_idx)
                )
                
                # generate a set of keyframes indexes for later use
                # keyframes_set.add((video_idx, sec))

                keyframe_boxes_and_labels[i].append([
                    instances[video_idx][sec]['gt_boxes'], 
                    instances[video_idx][sec]['gt_classes'], 
                    instances[video_idx][sec]['gt_actions'],
                    instances[video_idx][sec]['gt_ids_to_idxs'],
                    instances[video_idx][sec]['gt_idxs_to_ids'],
                ])
                if mode != 'train' and 'proposal_boxes' in instances[video_idx][sec]:
                    keyframe_boxes_and_labels[i][-1].extend([
                        instances[video_idx][sec]['proposal_boxes'],
                        instances[video_idx][sec]['proposal_classes'],
                        instances[video_idx][sec]['proposal_scores'],
                        # instances[video_idx][sec]['proposal_ids_to_idxs'],
                        # instances[video_idx][sec]['proposal_idxs_to_ids'],
                    ])
                    pred_count += 1

                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)
    logger.info("%d predicted keyframes used." % pred_count)

    # save a set of keyframes indexes for later use
    # import pickle
    # with open('keyframes_set.pkl', 'wb') as f:
    #     pickle.dump(keyframes_set, f)
    # with open('keyframes_set.pkl', 'rb') as f:
    #     keyframes_set2 = pickle.load(f)
    # import pdb; pdb.set_trace()

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count
