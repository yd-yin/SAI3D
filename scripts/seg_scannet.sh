#region growing on scannetpp
# source YOUR_CONDA_PATH/bin/activate opensai

#set the parameters below
VIEW_FREQ=5    # sample frequency of the frames
THRES_CONNECT="0.9,0.5,5"   # dynamic threshold for region growing
MAX_NEIGHBOR_DISTANCE=2    # farthest distance to take neighbors into account
THRES_MERGE=200           # merge small groups with less than THRES_MERGE points during post-processing
DIS_DECAY=0.5              # decay rate of the distance weight
SIMILAR_METRIC="2-norm"    # metric for similarity measurement
MASK_NAME="semantic-sam"   # mask name for loading mask
ALIAS_MASK_NAME="semantic-sam"   # mask name for saving results


TEXT_HEAD="demo_scannet"
TEXT="${TEXT_HEAD}_${VIEW_FREQ}view"
HEAD="${TEXT_HEAD}_${VIEW_FREQ}view_merge${THRES_MERGE}_${SIMILAR_METRIC}_${ALIAS_MASK_NAME}_connect(${THRES_CONNECT})_depth${MAX_NEIGHBOR_DISTANCE}"
EVAL_DIR="data/ScanNet/results/${HEAD}"   # directory to export results

python sai3d.py \
 --thres_merge=$THRES_MERGE \
 --similar_metric=$SIMILAR_METRIC \
 --thres_connect=$THRES_CONNECT \
 --mask_name=$MASK_NAME \
 --text=$TEXT \
 --max_neighbor_distance=$MAX_NEIGHBOR_DISTANCE \
 --view_freq=$VIEW_FREQ \
 --use_torch \
 --dis_decay=$DIS_DECAY \
 --eval_dir=$EVAL_DIR \