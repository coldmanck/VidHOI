#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# In[2]:


import os
import numpy as np
import json
from detectron2.structures import BoxMode

def get_vidor_dicts(dataset_dir, mode):
    json_file = os.path.join(dataset_dir, mode + "_frame_annots_detectron2.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        
        image_id = v['video_id'] + '_' + v['frame_id']
        filename = image_id + '.jpg'
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


# In[3]:


from detectron2.data import DatasetCatalog, MetadataCatalog

dataset_dir = os.path.join('slowfast', 'datasets', 'vidor')

with open(os.path.join(dataset_dir, 'obj_categories.json'), 'r') as f:
    obj_categories = json.load(f)

for d in ["train", "val"]:
    DatasetCatalog.register("vidor_" + d, lambda d=d: get_vidor_dicts(dataset_dir, d))
    MetadataCatalog.get("vidor_" + d).set(thing_classes=obj_categories)


# In[4]:


vidor_metadata = MetadataCatalog.get("vidor_train")
dataset_dicts = get_vidor_dicts(dataset_dir, 'train')


# In[17]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=vidor_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     # cv2.imshow(out.get_image()[:, :, ::-1])
#     plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()


# In[18]:


from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("vidor_train",)
cfg.DATASETS.TEST = ("",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.0003  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 78

def main(args):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


# In[19]:

# def main():
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
# trainer.train()

num_gpus = 8
# num_machines = 1
# machine_rank = 0

launch(trainer.train(), num_gpus)


# In[ ]:




