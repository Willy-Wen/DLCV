#!/bin/bash
# Original
# source scripts/train_models.sh Res16UNet34D 4 test
# source scripts/train_models.sh Res16UNet34D 2 test
export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

export WEIGHTS_SUFFIX=$5
# Original
export DATA_ROOT="/home/ynjuan/final-project-challenge-2-teamname/dataset"
# Add pretrain
# export DATA_ROOT="/home/ynjuan/final-project-challenge-2-teamname/dataset2"
export PRETRAINED_WEIGHTS="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/For-test/checkpoint-val_miou=23.17-step=1.ckpt"
export OUTPUT_DIR_ROOT="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$SUFFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"
# CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --weights $PRETRAINED_WEIGHTS \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize False \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 1 \
    --balanced_category_sampling True \
    $ARGS \
    2>&1 | tee -a "$LOG" \

# --weights $PRETRAINED_WEIGHTS \
# --resume $LOG_DIR \
