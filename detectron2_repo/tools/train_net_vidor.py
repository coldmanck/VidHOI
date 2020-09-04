#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

import json
from detectron2.structures import BoxMode
from detectron2 import model_zoo


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)

    # configs for training
    if args.small_vidor: # cfg.DATASETS.VIDOR.SIZE == 'small':
        cfg.DATASETS.TRAIN = ("vidor_small_train",)
    elif args.small_vidor_10imgs: # cfg.DATASETS.VIDOR.SIZE == 'small-10imgs':
        cfg.DATASETS.TRAIN = ("vidor_small_10imgs_train",)
    else:
        cfg.DATASETS.TRAIN = ("vidor_large_train",)
    # cfg.DATALOADER.NUM_WORKERS = 2
    if not args.eval_only:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    factor = 4
    cfg.SOLVER.IMS_PER_BATCH = 16 * factor
    cfg.SOLVER.BASE_LR = 0.0001 * factor # finetune using 10x smaller base_lr
    cfg.SOLVER.MAX_ITER = 270000 // factor 
    cfg.SOLVER.STEPS = [210000 // factor, 250000 // factor]
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default: 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 78

    # configs for testing
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if args.small_vidor: # cfg.DATASETS.VIDOR.SIZE == 'small':
        cfg.DATASETS.TEST = ("vidor_small_val",)
    elif args.small_vidor_10imgs: # cfg.DATASETS.VIDOR.SIZE == 'small-10imgs':
        cfg.DATASETS.TEST = ("vidor_small_10imgs_val",)
    else:
        cfg.DATASETS.TEST = ("vidor_large_val",)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # cfg.OUTPUT_DIR = './output/train_vidor_with_pseudo_labels'
    
    
    if not args.eval_only:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_vidor_dicts(dataset_dir, mode, small_vidor=False, small_vidor_10imgs=False):
    if small_vidor:
        json_file = os.path.join(dataset_dir, mode + "_frame_annots_detectron2_small_dataset.json")
    elif small_vidor_10imgs:
        json_file = os.path.join(dataset_dir, mode + "_frame_annots_detectron2_small_dataset_10imgs.json")
    else:
        json_file = os.path.join(dataset_dir, mode + "_frame_annots_detectron2.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        
        # image_id = v['video_id'] + '_' + v['frame_id']
        # filename = image_id + '.jpg'
        image_id = v['video_folder'] + '/' + v['video_id'] + '_' + v['frame_id']
        filename = v['video_id'] + '_' + v['frame_id'] + '.jpg'
        filename = os.path.join(dataset_dir, 'frames', v['video_folder'], v['video_id'], filename)
        
        record["file_name"] = filename
        record["image_id"] = image_id
        record["height"] = v['height']
        record["width"] = v['width']
      
        annos = v["objs"]
        objs = []
        for anno in annos:
            bbox = anno['bbox']
            obj = {
                "bbox": [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": anno['object_class'],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main(args):
    # register VidOR dataset
    dataset_dir = os.path.join('slowfast', 'datasets', 'vidor')
    with open(os.path.join(dataset_dir, 'obj_categories.json'), 'r') as f:
        obj_categories = json.load(f)

    cfg = setup(args)

    dataset_name = 'vidor'
    if args.small_vidor:
        dataset_name += '_small'
    elif args.small_vidor_10imgs:
        dataset_name += '_small_10imgs'
    else:
        dataset_name += '_large'
    # import pdb; pdb.set_trace()
    # small_vidor = small_vidor_10imgs = False
    # if cfg.DATASETS.VIDOR.SIZE == 'small':
    #     dataset_name += '_small'
    #     small_vidor = True
    # elif cfg.DATASETS.VIDOR.SIZE == 'small-10imgs':
    #     dataset_name += '_small_10imgs'
    #     small_vidor_10imgs = True
    # else:
    #     dataset_name += '_large'
    for d in ["train", "val"]:
        DatasetCatalog.register(dataset_name + '_' + d, lambda d=d: get_vidor_dicts(dataset_dir, d, args.small_vidor, args.small_vidor_10imgs))
        MetadataCatalog.get(dataset_name + '_' + d).set(thing_classes=obj_categories,evaluator_type='coco')
    # vidor_metadata = MetadataCatalog.get(dataset_name + '_train')
    # dataset_dicts = get_vidor_dicts(dataset_dir, 'train')

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
