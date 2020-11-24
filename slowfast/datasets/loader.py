#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler

from .build import build_dataset


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key in ["boxes", "ori_boxes"]:
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)
            
    return inputs, labels, video_idx, collated_extra_data


def hoi_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, extra_data = zip(*batch)
    inputs = default_collate(inputs)

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key in ["boxes", "ori_boxes", "obj_classes", "action_labels", "gt_boxes", "proposal_classes", "proposal_scores", "trajectories", "human_poses", "trajectory_boxes", "skeleton_imgs", "trajectory_box_masks"]:
            # Append idx info to the bboxes before concatenating them.
            if key in ['obj_classes', 'proposal_classes', 'proposal_scores']: # use a mask
                max_len = max([len(i) for i in data])
                # mask = np.zeros((len(data), max_len))
                obj_classes = []
                lengths = []
                for i in range(len(data)):
                    length = data[i].shape[0]
                    lengths.append(length)
                    entry = np.full((length, 1), -1)
                    entry[:] = float(i)
                    # entry = [np.full((data[i].shape[0], 1), float(i)), data[i].reshape(-1, 1)]
                    entry = np.concatenate((entry, data[i].reshape(-1, 1)), axis=1)
                    obj_classes.append(entry)
                    # mask[i][:length] = 1
                obj_classes = np.concatenate(obj_classes, axis=0)
                collated_extra_data[key] = torch.tensor(obj_classes).float()
                if key.startswith('proposal_'):
                    collated_extra_data['proposal_lengths'] = torch.tensor(lengths).int()
                else:
                    collated_extra_data[key + '_lengths'] = torch.tensor(lengths).int()
            elif key == 'action_labels':
                action_labels = []
                for i in range(len(data)):
                    n_bboxes = len(data[i])
                    entry = np.full((n_bboxes ** 2, 2), -1)
                    entry[:, 0] = float(i)
                    for j in range(n_bboxes):
                        entry[j * n_bboxes:(j+1) * n_bboxes, 1] = np.array(list(range(n_bboxes)))
                    entry = np.concatenate((entry, data[i].reshape(-1, 50)), axis=1)
                    action_labels.append(entry)
                action_labels = np.concatenate(action_labels, axis=0)
                collated_extra_data[key] = torch.tensor(action_labels).float()
            else:    
                try:
                    if key in ['skeleton_imgs', 'trajectory_box_masks']:
                        bboxes = [
                            np.concatenate(
                                [np.full((data[i].shape[0], 1, data[i].shape[2], data[i].shape[3]), float(i)), data[i]], axis=1
                            )
                            for i in range(len(data))
                        ]
                    else:
                        bboxes = [
                            np.concatenate(
                                [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                            )
                            for i in range(len(data))
                        ]
                except:
                    import pdb; pdb.set_trace()
                bboxes = np.concatenate(bboxes, axis=0)
                collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)
            
    return inputs, collated_extra_data


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE) if cfg.DETECTION.ENABLE_HOI else int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE)
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
        # Create a sampler for multi-process training
        sampler = (
            DistributedSampler(dataset)
            if cfg.NUM_GPUS > 1
            else RandomSampler(dataset)
        )
        batch_sampler = ShortCycleBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
        )
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            collate_fn=hoi_collate if cfg.DETECTION.ENABLE_HOI else (detection_collate if cfg.DETECTION.ENABLE else None),
        )
    else:
        # Create a sampler for multi-process training
        sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=hoi_collate if cfg.DETECTION.ENABLE_HOI else (detection_collate if cfg.DETECTION.ENABLE else None),
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = (
        loader.batch_sampler.sampler
        if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
        else loader.sampler
    )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
