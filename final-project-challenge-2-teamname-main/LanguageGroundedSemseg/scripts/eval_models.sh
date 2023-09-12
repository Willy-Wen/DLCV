#!/bin/bash
# source scripts/eval_models.sh Res16UNet34D 1 visual-2
export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/home/ynjuan/final-project-challenge-2-teamname/for-test-dataset"
export PRETRAINED_WEIGHTS="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/For-test/checkpoint-val_miou=23.17-step=1.ckpt"
export OUTPUT_DIR_ROOT="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$SUFFIX
export MODEL_WEIGHTS="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/For-test"
export VISUALIZE_PATH="/nfs/nas-6.1/ynjuan/v2"
# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"
# CUDA_VISIBLE_DEVICES=1 
# 在做 evaluation 的時候，只能使用一個 GPU!
# 最好也只用一個 batch size
CUDA_VISIBLE_DEVICES=1 python -m my_main \
    --is_train False \
    --weights $PRETRAINED_WEIGHTS \
    --save_prediction True \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize True \
    --visualize_path  $VISUALIZE_PATH/visualize \
    --num_gpu 2 \
    --balanced_category_sampling True \
    $ARGS \
    2>&1 | tee -a "$LOG" \
#    --weights $PRETRAINED_WEIGHTS \