#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# ADAPTED FROM OPENMASK3D SCANNET200 EVALUATION SCRIPT
# This script performs the following in order to evaluate OpenMask3D predictions on the ScanNet200 validation set--------------------------------------
# 1. Compute mask features for each mask and save them
# 2. Evaluate for closed-set 3D semantic instance segmentation

# --------
# NOTE: SET THESE PARAMETERS!
BASE_DIR="/code/SAI3D_test"  # abs path to the SAI3D directory
SCANS_PATH="${BASE_DIR}/data/ScanNet_OpenMask3D"        # abs path to the ScanNet dataset of the OpenMask3D format
# model ckpt paths
SAM_CKPT_PATH="/openmask3d/checkpoints/segment-anything/sam_vit_h_4b8939.pth"  #abs path to the SAM model checkpoint
# output directories to save masks and mask features
EXPERIMENT_NAME="scannet200"
OUTPUT_FOLDER_DIRECTORY="${BASE_DIR}/output"  # dir to save hydra log
MASK_SAVE_DIR="${BASE_DIR}/data/class_agnostic_masks" # abs path to the dir saving the class-agnostic masks with OpenMask3D format
MASK_FEATURE_SAVE_DIR="${BASE_DIR}/data/mask_features" # abs path to the dir saving the mask features calculated by OpenMask3D
SCANNET_INSTANCE_GT_DIR="${BASE_DIR}/data/gt_scannet200"
# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd openmask3d/openmask3d

# 1. Compute mask features
echo "[INFO] Computing mask features..."
python compute_features_scannet200.py \
data.scans_path=${SCANS_PATH} \
data.masks.masks_path=${MASK_SAVE_DIR} \
output.output_directory=${MASK_FEATURE_SAVE_DIR} \
output.experiment_name=${EXPERIMENT_NAME} \
external.sam_checkpoint=${SAM_CKPT_PATH} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
echo "[INFO] Feature computation done!"

# 2. Evaluate for closed-set 3D semantic instance segmentation
python evaluation/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--mask_pred_dir=${MASK_SAVE_DIR} \
--mask_features_dir=${MASK_FEATURE_SAVE_DIR} \
