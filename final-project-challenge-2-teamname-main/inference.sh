#!/bin/bash
# source scripts/eval_models.sh Res16UNet34D 1 visual-2
# ./inference.sh /home/ynjuan/final-project-challenge-2-teamname/test ./output
export PYTHONUNBUFFERED="True"
export PYTHONPATH='./LanguageGroundedSemseg'

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=Res16UNet34D
export BATCH_SIZE=1
export PLY_PATH=$1
export TXT_PATH=$2
export DATA_ROOT="./my_test_dataset"
export PRETRAINED_WEIGHTS="./ckpt/model.ckpt"
# export PRETRAINED_WEIGHTS="/home/ynjuan/final-project-challenge-2-teamname/LanguageGroundedSemseg/ckpt/down-stream/checkpoint-val_miou=26.07-step=8981.ckpt"
export LOG_DIR=$3
export VISUALIZE_PATH="./tmp"

# CUDA_VISIBLE_DEVICES=1 
# 在做 evaluation 的時候，只能使用一個 GPU!
# 最好也只用一個 batch size
python -m my_main \
    --is_train False \
    --weights $PRETRAINED_WEIGHTS \
    --save_prediction True \
    --log_dir $LOG_DIR \
    --ply_path $PLY_PATH \
    --txt_path $TXT_PATH \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize True \
    --visualize_path  $VISUALIZE_PATH/visualize \
    --num_gpu 1 \
    --balanced_category_sampling True \

rm -r -f $VISUALIZE_PATH/visualize
rm -r -f $LOG_DIR/lightning_logs