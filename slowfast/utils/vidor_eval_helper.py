# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# ActivityNet
# Copyright (c) 2015 ActivityNet
# Licensed under The MIT License
# [see https://github.com/activitynet/ActivityNet/blob/master/LICENSE for details]
# --------------------------------------------------------

"""Helper functions for AVA evaluation."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import csv
import logging
import numpy as np
import pprint
import time
from collections import defaultdict
from fvcore.common.file_io import PathManager

from slowfast.utils.ava_evaluation import (
    object_detection_evaluation,
    standard_fields,
)

import pickle
import os, json
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv_vidor(csv_file, class_whitelist=None, load_score=False):
    """Loads boxes and class labels from a CSV file in the AVA format.
    CSV file format described at https://research.google.com/ava/download.html.
    Args:
      csv_file: A file object.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.
    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    with PathManager.open(csv_file, "r") as f:
        reader = csv.reader(f)

        import pdb; pdb.set_trace()
        for row in reader:
            assert len(row) in [7, 8], "Wrong number of columns: " + row
            image_key = make_image_key(row[0], row[1])
            x1, y1, x2, y2 = [float(n) for n in row[2:6]]
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue
            score = 1.0
            if load_score:
                score = float(row[7])
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    return boxes, labels, scores


def read_exclusions_vidor(exclusions_file):
    """Reads a CSV file of excluded timestamps.
    Args:
      exclusions_file: A file object containing a csv of video-id,timestamp.
    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        with PathManager.open(exclusions_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2, "Expected only 2 columns, got: " + row
                excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap_vidor(labelmap_file):
    """Read label map and class ids."""

    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""

    with open('idx_to_pred.pkl', 'rb') as f:
        idx_to_pred = pickle.load(f)

    # with PathManager.open(labelmap_file, "r") as f:
    #     import pdb; pdb.set_trace()
    #     for line in f:
    #         if line.startswith("  name:"):
    #             name = line.split('"')[1]
    #         elif line.startswith("  id:") or line.startswith("  label_id:"):
    #             class_id = int(line.strip().split(" ")[-1])
    #             labelmap.append({"id": class_id, "name": name})
    #             class_ids.add(class_id)
    # return labelmap, class_ids

    """
    (Pdb) categories
    [{'id': 1, 'name': 'bend/bow (at the waist)'}, {'id': 3, 'name': 'crouch/kneel'}, {'id': 4, 'name': 'dance'}, {'id': 5, 'name': 'fall down'}, {'id': 6, 'name': 'get up'}, {'id': 7, 'name': 'jump/leap'}, {'id': 8, 'name': 'lie/sleep'}, {'id': 9, 'name': 'martial art'}, {'id': 10, 'name': 'run/jog'}, {'id': 11, 'name': 'sit'}, {'id': 12, 'name': 'stand'}, {'id': 13, 'name': 'swim'}, {'id': 14, 'name': 'walk'}, {'id': 15, 'name': 'answer phone'}, {'id': 17, 'name': 'carry/hold (an object)'}, {'id': 20, 'name': 'climb (e.g., a mountain)'}, {'id': 22, 'name': 'close (e.g., a door, a box)'}, {'id': 24, 'name': 'cut'}, {'id': 26, 'name': 'dress/put on clothing'}, {'id': 27, 'name': 'drink'}, {'id': 28, 'name': 'drive (e.g., a car, a truck)'}, {'id': 29, 'name': 'eat'}, {'id': 30, 'name': 'enter'}, {'id': 34, 'name': 'hit (an object)'}, {'id': 36, 'name': 'lift/pick up'}, {'id': 37, 'name': 'listen (e.g., to music)'}, {'id': 38, 'name': 'open (e.g., a window, a car door)'}, {'id': 41, 'name': 'play musical instrument'}, {'id': 43, 'name': 'point to (an object)'}, {'id': 45, 'name': 'pull (an object)'}, {'id': 46, 'name': 'push (an object)'}, {'id': 47, 'name': 'put down'}, {'id': 48, 'name': 'read'}, {'id': 49, 'name': 'ride (e.g., a bike, a car, a horse)'}, {'id': 51, 'name': 'sail boat'}, {'id': 52, 'name': 'shoot'}, {'id': 54, 'name': 'smoke'}, {'id': 56, 'name': 'take a photo'}, {'id': 57, 'name': 'text on/look at a cellphone'}, {'id': 58, 'name': 'throw'}, {'id': 59, 'name': 'touch (an object)'}, {'id': 60, 'name': 'turn (e.g., a screwdriver)'}, {'id': 61, 'name': 'watch (e.g., TV)'}, {'id': 62, 'name': 'work on a computer'}, {'id': 63, 'name': 'write'}, {'id': 64, 'name': 'fight/hit (a person)'}, {'id': 65, 'name': 'give/serve (an object) to (a person)'}, {'id': 66, 'name': 'grab (a person)'}, {'id': 67, 'name': 'hand clap'}, {'id': 68, 'name': 'hand shake'}, {'id': 69, 'name': 'hand wave'}, {'id': 70, 'name': 'hug (a person)'}, {'id': 72, 'name': 'kiss (a person)'}, {'id': 73, 'name': 'lift (a person)'}, {'id': 74, 'name': 'listen to (a person)'}, {'id': 76, 'name': 'push (another person)'}, {'id': 77, 'name': 'sing to (e.g., self, a person, a group)'}, {'id': 78, 'name': 'take (an object) from (a person)'}, {'id': 79, 'name': 'talk to (e.g., self, a person, a group)'}, {'id': 80, 'name': 'watch (a person)'}]
    (Pdb) class_whitelist
    {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24, 26, 27, 28, 29, 30, 34, 36, 37, 38, 41, 43, 45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80}
    """


# def evaluate_ava_from_files(labelmap, groundtruth, detections, exclusions):
#     """Run AVA evaluation given annotation/prediction files."""

#     categories, class_whitelist = read_labelmap(labelmap)
#     excluded_keys = read_exclusions(exclusions)
#     groundtruth = read_csv(groundtruth, class_whitelist, load_score=False)
#     detections = read_csv(detections, class_whitelist, load_score=True)
#     run_evaluation(categories, groundtruth, detections, excluded_keys)

def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

def evaluate_vidor(
    all_preds,
    all_preds_score,
    all_preds_bbox_pair_ids,
    all_proposal_scores,
    all_proposal_boxes,
    all_proposal_classes,
    all_gt_boxes,
    all_gt_action_labels,
    all_gt_obj_classes,
    all_gt_bbox_pair_ids,
):
    logger.info(f'Number of validation images: {len(all_preds)}')

    tp, fp, scores, sum_gt = {}, {}, {}, {}

    # Construct dictionaries of triplet class 
    for idx in tqdm(range(len(all_preds))):
        bbox_pair_ids = all_gt_bbox_pair_ids[idx]
        
        for jdx, action_label in enumerate(all_gt_action_labels[idx]):
            # action_label != 0.0
            action_label_idxs = [i for i in range(50) if action_label[i] == 1.0]
            for action_label_idx in action_label_idxs:
                subject_label_idx = 0 # person idx = 0
                # print(idx, len(all_gt_obj_classes))
                # import pdb; pdb.set_trace()
                object_label_idx = int(all_gt_obj_classes[idx][bbox_pair_ids[jdx][1]][1])
                triplet_class = (subject_label_idx, action_label_idx, object_label_idx)
                
                if triplet_class not in tp: # should also not exist in fp, scores & sum_gt
                    tp[triplet_class] = []
                    fp[triplet_class] = []
                    scores[triplet_class] = []
                    sum_gt[triplet_class] = 0
                sum_gt[triplet_class] += 1

    # Collect true positive, false positive & scores
    correct_det_count = correct_hoi_count = total_det_count = 0
    at_least_one_pair_bbox_detected_count = total_count = 0
    for idx in tqdm(range(len(all_preds))): # for each keyframe
        preds_bbox_pair_ids = all_preds_bbox_pair_ids[idx]
        gt_bbox_pair_ids = all_gt_bbox_pair_ids[idx]
        gt_bbox_pair_matched = set()
        
        # take only top 100 confident triplets
        preds_scores = [
            (math.log(all_preds_score[idx][i][j] if all_preds_score[idx][i][j] > 0 else 1e-300) + \
            math.log(all_proposal_scores[idx][preds_bbox_pair_ids[i][0]][1] if all_proposal_scores[idx][preds_bbox_pair_ids[i][0]][1] > 0 else 1e-300) + \
            math.log(all_proposal_scores[idx][preds_bbox_pair_ids[i][1]][1] if all_proposal_scores[idx][preds_bbox_pair_ids[i][1]][1] > 0 else 1e-300), i, j) 
            for i in range(len(all_preds_score[idx])) for j in range(len(all_preds_score[idx][i]))
        ]
        preds_scores.sort(reverse=True)
        preds_scores = preds_scores[:100]
        
        at_least_one_pair_bbox_detected = False
        
        for score, i, j in preds_scores: # for each HOI prediction, i-th pair and j-th action
            pred_sub_cls = int(all_proposal_classes[idx][preds_bbox_pair_ids[i][0]][1])
            pred_obj_cls = int(all_proposal_classes[idx][preds_bbox_pair_ids[i][1]][1])
            pred_rel_cls = j
            
            triplet_class = (pred_sub_cls, pred_rel_cls, pred_obj_cls)
            if triplet_class not in tp:
                continue
            
            pred_sub_box = all_proposal_boxes[idx][preds_bbox_pair_ids[i][0]][1:]
            pred_obj_box = all_proposal_boxes[idx][preds_bbox_pair_ids[i][1]][1:]
            is_match = False
            max_ov = max_gt_id = 0
            for k, gt_bbox_pair_id in enumerate(gt_bbox_pair_ids):  # for each ground truth HOI
                gt_sub_cls = int(all_gt_obj_classes[idx][gt_bbox_pair_id[0]][1])
                gt_obj_cls = int(all_gt_obj_classes[idx][gt_bbox_pair_id[1]][1])
                gt_rel_cls = all_gt_action_labels[idx][k][j]
                
                gt_sub_box = all_gt_boxes[idx][gt_bbox_pair_id[0]][1:]
                gt_obj_box = all_gt_boxes[idx][gt_bbox_pair_id[1]][1:]
                sub_ov = bbox_iou(gt_sub_box, pred_sub_box)
                obj_ov = bbox_iou(gt_obj_box, pred_obj_box)

                if gt_sub_cls == pred_sub_cls and gt_obj_cls == pred_obj_cls and sub_ov >= 0.5 and obj_ov >= 0.5:
                    correct_det_count += 1
                    if not at_least_one_pair_bbox_detected:
                        at_least_one_pair_bbox_detected = True
                        at_least_one_pair_bbox_detected_count += 1
                        
                    if gt_rel_cls == 1.0:
                        is_match = True
                        correct_hoi_count += 1
                        min_ov_cur = min(sub_ov, obj_ov)
                        if min_ov_cur > max_ov:
                            max_ov = min_ov_cur
                            max_gt_id = k
                total_det_count += 1
            
            if is_match and max_gt_id not in gt_bbox_pair_matched:
                tp[triplet_class].append(1)
                fp[triplet_class].append(0)
                gt_bbox_pair_matched.add(max_gt_id)
            else:
                tp[triplet_class].append(0)
                fp[triplet_class].append(1)
            scores[triplet_class].append(score)
        total_count += 1

    if correct_det_count != 0:
        hd = correct_hoi_count/correct_det_count
    else:
        hd = 0
        logger.info('correct_det_count is 0!')
    if total_det_count != 0:
        dt = correct_det_count/total_det_count
    else: 
        dt = 0
        logger.info('total_det_count is 0!')
    if total_count != 0:
        one_dr = at_least_one_pair_bbox_detected_count/total_count
    else:
        one_dr = 0
        logger.info('total_count is 0!')
    logger.info(f'[Final] correct HOI/correct detection ratio: {hd:5f}')
    logger.info(f'[Final] correct detection/total detection ratio: {dt:5f}')
    logger.info(f'[Final] at_least_one_pair_bbox_detected ratio: {at_least_one_pair_bbox_detected_count} / {total_count} = {one_dr:5f}')

    # Compute mAP & recall
    ap = np.zeros(len(tp))
    max_recall = np.zeros(len(tp))

    for i, triplet_class in tqdm(enumerate(tp.keys())):
        sum_gt_temp = sum_gt[triplet_class]

        if sum_gt_temp == 0:
            continue
        tp_temp = np.asarray((tp[triplet_class]).copy())
        fp_temp = np.asarray((fp[triplet_class]).copy())
        res_num = len(tp_temp)
        if res_num == 0:
            continue
        scores_temp = np.asarray(scores[triplet_class].copy())
        sort_inds = np.argsort(-scores_temp) # sort decreasingly

        fp_temp = fp_temp[sort_inds]
        tp_temp = tp_temp[sort_inds]

        fp_temp = np.cumsum(fp_temp) 
        tp_temp = np.cumsum(tp_temp)

        rec = tp_temp / sum_gt_temp # recall
        prec = tp_temp / (fp_temp + tp_temp) # precision

        ap[i] = voc_ap(rec, prec)
        max_recall[i] = np.max(rec)

    mAP = np.mean(ap[:])
    m_rec = np.mean(max_recall[:])

    logger.info(f'[Final] mAP: {mAP}')
    logger.info(f'[Final] max recall: {m_rec}')

    return mAP, m_rec, hd, dt, one_dr
    

# def evaluate_vidor(
#     preds,
#     original_boxes,
#     metadata,
#     excluded_keys,
#     class_whitelist,
#     categories,
#     groundtruth=None,
#     video_idx_to_name=None,
#     name="latest",
# ):
#     """Run AVA evaluation given numpy arrays."""

#     eval_start = time.time()

#     detections = get_vidor_eval_data(
#         preds,
#         original_boxes,
#         metadata,
#         class_whitelist,
#         video_idx_to_name=video_idx_to_name,
#     )

#     logger.info("Evaluating with %d unique GT frames." % len(groundtruth[0]))
#     logger.info(
#         "Evaluating with %d unique detection frames" % len(detections[0])
#     )

#     write_results(detections, "detections_%s.csv" % name)
#     write_results(groundtruth, "groundtruth_%s.csv" % name)

#     results = run_evaluation(categories, groundtruth, detections, excluded_keys)

#     logger.info("AVA eval done in %f seconds." % (time.time() - eval_start))
#     return results["PascalBoxes_Precision/mAP@0.5IOU"]


def run_evaluation(
    categories, groundtruth, detections, excluded_keys, verbose=True
):
    """AVA evaluation main logic."""

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories
    )

    boxes, labels, _ = groundtruth

    gt_keys = []
    pred_keys = []

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in ground truth: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

        gt_keys.append(image_key)

    boxes, labels, scores = detections

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in detections: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

        pred_keys.append(image_key)

    metrics = pascal_evaluator.evaluate()

    pprint.pprint(metrics, indent=2)
    return metrics


def get_vidor_eval_data(
    scores,
    boxes,
    metadata,
    class_whitelist,
    verbose=False,
    video_idx_to_name=None,
):
    """
    Convert our data format into the data format used in official AVA
    evaluation.
    """

    out_scores = defaultdict(list)
    out_labels = defaultdict(list)
    out_boxes = defaultdict(list)
    count = 0
    for i in range(scores.shape[0]):
        video_idx = int(np.round(metadata[i][0]))
        sec = int(np.round(metadata[i][1]))

        video = video_idx_to_name[video_idx]

        key = video + "," + "%04d" % (sec)
        batch_box = boxes[i].tolist()
        # The first is batch idx.
        batch_box = [batch_box[j] for j in [0, 2, 1, 4, 3]]

        one_scores = scores[i].tolist()
        for cls_idx, score in enumerate(one_scores):
            if cls_idx + 1 in class_whitelist:
                out_scores[key].append(score)
                out_labels[key].append(cls_idx + 1)
                out_boxes[key].append(batch_box[1:])
                count += 1

    return out_boxes, out_labels, out_scores


def write_results(detections, filename):
    """Write prediction results into official formats."""
    start = time.time()

    boxes, labels, scores = detections
    with PathManager.open(filename, "w") as f:
        for key in boxes.keys():
            for box, label, score in zip(boxes[key], labels[key], scores[key]):
                f.write(
                    "%s,%.03f,%.03f,%.03f,%.03f,%d,%.04f\n"
                    % (key, box[1], box[0], box[3], box[2], label, score)
                )

    logger.info("AVA results wrote to %s" % filename)
    logger.info("\ttook %d seconds." % (time.time() - start))
