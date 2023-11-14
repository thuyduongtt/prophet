#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    ;;
  2)
    DS_NAME="balanced_10"
    ;;
esac

# DATA PREPARATION - Step 2
# bash scripts/extract_img_feats.sh --dataset $DS_NAME

# STAGE ONE
# Step 4
# bash scripts/finetune.sh --task $DS_NAME --version ${DS_NAME}_finetune_1 --pretrained_model ckpts/mcan_pt_okvqa.pkl

# Step 5
bash scripts/heuristics_gen.sh \
    --version ${DS_NAME}_heuristics_1
    --ckpt_path outputs/${DS_NAME}_finetune_1/ckpts/epoch_6.pkl
    --candidate_num 10 --example_num 100