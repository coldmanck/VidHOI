# ST-HOI

This repo includes models and experiment codes of "ST-HOI: A Spatial-Temporal Baseline for Human Object Interaction in Videos" (paper will be released soon).

## Installation
```
conda create -n slowfast python=3.6 scipy numpy
conda activate slowfast
# Install PyTorch 1.4.0 and torchvision 0.5.0 first
pip install -r requirements.txt
# Another way
# conda create --name slowfast --file requirements.txt

# Then refer to OLD_README.md to install SlowFast and detectron2.
```

## Dataset
VidOR dataset is used. One may download the dataset and the original annotation at [the official website](https://xdshang.github.io/docs/vidor.html) and unzip to `$ROOT/slowfast/dataset/vidor`. For HOI-specific annotations, refer to files under the same folder, and for larger files, download from [here](https://drive.google.com/drive/folders/1PGZ-5vGXphL5dgUWrlePZn5lQ2ejq62K?usp=sharing).

One then needs to extract frames from VidOR videos using `$ROOT/slowfast/dataset/vidor/extract_vidor_frames.sh`.

## Experiments
For checking each model's final performance including mAP, use `$ROOT/vidor_eval.ipynb`. The following commands use ground truth `GT` (Oracle mode) by default. To use detected trajectories, refer to `NONGT` version of each model.

### Image Baseline (2D Model)
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_GT.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_GT.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_GT/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `BASELINE_32x2_R50_SHORT_SCRATCH_EVAL_NONGT`

### Video Baseline (3D Model)
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_NONGT`

### 3D + Trajectory (Ours-T)
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_NONGT_trajectory`

### 3D + Trajectory + ToI Features (Ours-T+V)
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_NONGT_trajectory-toipool`

### 3D + Trajectory + Masking Pose Feature (Ours-T+P)
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-spa_conf.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 96 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-spa_conf.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-spa_conf/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_NONGT_trajectory-spa_conf`

### 3D + Trajectory + ToI Features + Masking Pose Feature (Ours-T+V+P)
Note that batch size is 112 for the this model.
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool-spa_conf.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 7 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 112 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool-spa_conf.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool-spa_conf/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```
- NON-GT version: `SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_NONGT_trajectory-toipool-spa_conf`


## Optional Experiments

### 3D + Trajectory + Human Poses (from VIBE)
#### [Pre-requisite]
Run VIBE with the following commands at `~/VIBE` to generate human poses for VidOR dataset and move the generated pose to the vidor dataset folder:
- `python demo_vidor.py --output_folder output/ --gt_tracklet --mode training`
- `python demo_vidor.py --output_folder output/ --gt_tracklet --mode validation`

Then run the following:
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-human_pose.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-human_pose.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-human_pose/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```

### Video baseline + Trajectory + Relativity Feature
- Training: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_relativity-feat.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 8 DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False VIDOR.TEST_DEBUG False
```
- Validation: Run 
```
python tools/run_net_vidor.py --cfg configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_relativity-feat.yaml DATA.PATH_TO_DATA_DIR slowfast/datasets/vidor NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 0 TEST.BATCH_SIZE 1 LOG_MODEL_INFO False TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH ./output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_relativity-feat/checkpoints/checkpoint_epoch_00020.pyth TRAIN.CHECKPOINT_TYPE pytorch VIDOR.TEST_DEBUG False
```

## Human Poses Inference from HRNet
```
# Clone the HRNet repo first
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git
# And download corresponding pretrained model weights
```
### Original inference command
```
python inference.py --cfg inference-config.yaml --videoFile ../videos/3418738633.mp4 --writeBoxFrames --outputDir output TEST.MODEL_FILE ../models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth
```
### Revised inference script with line graph
```
python inference_sticks.py --cfg inference-config_w48.yaml --videoFile ../videos/3418738633.mp4 --writeBoxFrames --outputDir output/w48 TEST.MODEL_FILE ../models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth
```
- check `args.inferenceFps`
### Revised inference script with line graph with ground truth person bbox
```
python inference_sticks_vidor_demo.py --cfg inference-config_w48.yaml --writeBoxFrames --outputDir output/vidor/w48 --output_video TEST.MODEL_FILE ../models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth
```
- set up video idx etc. inside the script
### Another way for inference with gt person bbox
```
python tools/inference_vidor.py --cfg experiments/vidor/hrnet/w48_384x288_adam_lr1e-3.yaml DATASET.SPLIT training
```
Replace `training` to `validation` for generating human poses for val split.