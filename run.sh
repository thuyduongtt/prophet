#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    ;;
  2)
    DS_NAME="balanced_10"
    ;;
esac

python test_gpu.py

# DATA PREPARATION - Step 2
# bash scripts/extract_img_feats.sh --dataset $DS_NAME

# STAGE ONE - Step 4
bash scripts/finetune.sh --version ${DS_NAME}_finetune_1 --pretrained_model ckpts/mcan_pt_okvqa.pkl