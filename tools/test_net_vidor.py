#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import VidorMeter

logger = logging.get_logger(__name__)

from tqdm import tqdm
import json
import pickle
import torch.nn.functional as F

@torch.no_grad()
def perform_test(test_loader, model, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    # test_meter.iter_tic()

    all_results = []
    # all_preds = []
    # all_action_labels = []
    # all_boxes = []
    # all_obj_classes = []
    # all_obj_classes_lengths = []

    count = 0
    use_proposal = False
    if cfg.DEMO.ENABLE:
        assert cfg.DEMO.VIDEO_IDX != ''
        found = False    
    for inputs, meta in tqdm(test_loader):
        if cfg.DEMO.ENABLE:
            # import pdb; pdb.set_trace()
            if meta['orig_video_idx'][0] != cfg.DEMO.VIDEO_IDX:
                print(meta['orig_video_idx'][0])
                if found:
                    break
                continue
            found = True
            meta.pop('orig_video_idx')
        elif cfg.DEMO.ENABLE_ALL:
            orig_video_idx = meta['orig_video_idx']
            meta.pop('orig_video_idx')

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Compute the predictions.
        # import pdb; pdb.set_trace()
        if 'proposal_classes' in meta:
            use_proposal = True
        # preds, action_labels, bbox_pair_ids = model(inputs, meta["boxes"], meta['obj_classes'], meta['obj_classes_lengths'], meta['action_labels'])
        
        trajectories = human_poses = trajectory_boxes = skeleton_imgs = trajectory_box_masks = None
        if cfg.MODEL.USE_TRAJECTORIES:
            trajectories = meta['trajectories']
        if cfg.MODEL.USE_HUMAN_POSES:
            human_poses = meta['human_poses']
        if cfg.DETECTION.ENABLE_TOI_POOLING or cfg.MODEL.USE_TRAJECTORY_CONV:
            trajectory_boxes = meta['trajectory_boxes']
        if cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = meta['skeleton_imgs']
            trajectory_box_masks = meta['trajectory_box_masks']
        preds, action_labels, bbox_pair_ids, gt_bbox_pair_ids = model(inputs, meta["boxes"], meta['proposal_classes'], meta['proposal_lengths'], meta['action_labels'], meta['obj_classes'], meta['obj_classes_lengths'], trajectories=trajectories, human_poses=human_poses, trajectory_boxes=trajectory_boxes, skeleton_imgs=skeleton_imgs, trajectory_box_masks=trajectory_box_masks)

        preds_score = F.sigmoid(preds).cpu()
        preds = preds_score >= 0.5 # Convert scores into 'True' or 'False'
        action_labels = action_labels.cpu()
        boxes = meta["boxes"].cpu()
        obj_classes = meta['obj_classes'].cpu()
        # obj_classes_lengths = meta['obj_classes_lengths'].cpu()
        bbox_pair_ids = bbox_pair_ids.cpu()
        gt_bbox_pair_ids = gt_bbox_pair_ids.cpu()
        # hopairs = hopairs # .cpu()
        proposal_scores = meta['proposal_scores'].cpu()
        gt_boxes = meta['gt_boxes'].cpu()
        proposal_classes = meta['proposal_classes'].cpu()

        if cfg.NUM_GPUS > 1:
            preds_score = torch.cat(du.all_gather_unaligned(preds_score), dim=0)
            preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
            action_labels = torch.cat(du.all_gather_unaligned(action_labels), dim=0)
            boxes = torch.cat(du.all_gather_unaligned(boxes), dim=0)
            obj_classes = torch.cat(du.all_gather_unaligned(obj_classes), dim=0)
            # obj_classes_lengths = torch.cat(du.all_gather_unaligned(obj_classes_lengths), dim=0),
            bbox_pair_ids = torch.cat(du.all_gather_unaligned(bbox_pair_ids), dim=0)
            gt_bbox_pair_ids = torch.cat(du.all_gather_unaligned(gt_bbox_pair_ids), dim=0)
            # hopairs = torch.cat(du.all_gather_unaligned(hopairs), dim=0)
            proposal_scores = torch.cat(du.all_gather_unaligned(proposal_scores), dim=0)
            gt_boxes = torch.cat(du.all_gather_unaligned(gt_boxes), dim=0)
            proposal_classes = torch.cat(du.all_gather_unaligned(proposal_classes), dim=0)

        results = {
            'preds_score': preds_score.detach().cpu().tolist(),
            'preds': preds.detach().cpu().tolist(),
            'preds_bbox_pair_ids': bbox_pair_ids.detach().cpu().tolist(),
            'proposal_scores': proposal_scores.detach().cpu().tolist(),
            'proposal_boxes': boxes.detach().cpu().tolist(),
            'proposal_classes': proposal_classes.detach().cpu().tolist(),
            'gt_boxes': gt_boxes.detach().cpu().tolist(),
            'gt_action_labels': action_labels.detach().cpu().tolist(),
            'gt_obj_classes': obj_classes.detach().cpu().tolist(),
            'gt_bbox_pair_ids': gt_bbox_pair_ids.detach().cpu().tolist(),
            # 'obj_classes_lengths': obj_classes_lengths.detach().cpu().tolist(),
            # 'hopairs': hopairs.detach().cpu().tolist() if isinstance(hopairs, torch.Tensor) else hopairs
        }
        
        if cfg.DEMO.ENABLE_ALL:
            results['orig_video_idx'] = orig_video_idx

        all_results.append(results)

        # import pdb; pdb.set_trace()

        count += 1
        if cfg.VIDOR.TEST_DEBUG and count == 10: # for debug purpose only!
            break

    # Log epoch stats and print the final testing results.
    '''
    if writer is not None:
        all_preds_cpu = [
            pred.clone().detach().cpu() for pred in test_meter.video_preds
        ]
        all_labels_cpu = [
            label.clone().detach().cpu() for label in test_meter.video_labels
        ]
        writer.plot_eval(preds=all_preds_cpu, labels=all_labels_cpu)
    '''
    # import pdb; pdb.set_trace()
    save_file_name = cfg.OUTPUT_DIR + '/all_results_vidor_' + cfg.TEST.CHECKPOINT_FILE_PATH.split('/')[-1]
    if cfg.VIDOR.TEST_DEBUG:
        save_file_name += '_debug'
    if use_proposal:
        save_file_name += '_proposal'
    if cfg.VIDOR.TEST_GT_LESS_TO_ALIGN_NONGT:
        save_file_name += '_less-168-examples'
    if cfg.DEMO.ENABLE:
        save_file_name += '_demo' + '_'.join(cfg.DEMO.VIDEO_IDX.split('/'))
    elif cfg.DEMO.ENABLE_ALL:
        save_file_name += '_demo-all'
    save_file_name += '.json'

    print(f'Saving to {save_file_name}')
    with open(save_file_name, 'w') as f:
        json.dump(all_results, f)
    print('Done!')

    # test_meter.finalize_metrics()
    # test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    
    if cfg.DEMO.ENABLE and cfg.DEMO.SPLIT == 'train':
        test_loader = loader.construct_loader(cfg, "train")
    else:
        test_loader = loader.construct_loader(cfg, "test")

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
    # test_meter = VidorMeter(len(test_loader), cfg, mode="test")

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, cfg, writer)
    if writer is not None:
        writer.close()
