#!/bin/bash
# source scripts/text_representation_train.sh 4 train-pretrain
# Exit script when a command returns nonzero state
export PYTHONUNBUFFERED="True"
export PYTHONPATH='./LanguageGroundedSemseg'

export BATCH_SIZE=4
export MODEL=Res16UNet34D
export DATASET=Scannet200Textual2cmDataset

export DATA_ROOT="./my_train_dataset"
export OUTPUT_DIR_ROOT="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"
python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --ply_path $1 \
    --txt_path $2 \
    --loss_type cross_entropy \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --train_limit_numpoints 1400000 \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --num_gpu 2 \
    --balanced_category_sampling False \
    --use_embedding_loss True \
    $ARGS \
    2>&1 | tee -a "$LOG"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize False \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 2 \
    --balanced_category_sampling True \
    $ARGS \
    2>&1 | tee -a "$LOG" \
