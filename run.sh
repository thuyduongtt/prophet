#!/bin/bash

N=2

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
# Step 3
bash scripts/pretrain.sh --task $DS_NAME --version ${DS_NAME}_pretrain_${$EXP}

# Step 4
# bash scripts/finetune.sh --task $DS_NAME --version ${DS_NAME}_finetune_${$EXP} --pretrained_model ckpts/mcan_pt_okvqa.pkl
bash scripts/finetune.sh \
 --task $DS_NAME \
 --version ${DS_NAME}_finetune_${$EXP} \
 --pretrained_model outputs/ckpts/${DS_NAME}_pretrain_${$EXP}/epoch13.pkl

# Step 5
bash scripts/heuristics_gen.sh \
    --task $DS_NAME \
    --version ${DS_NAME}_heuristics_${$EXP} \
    --ckpt_path outputs/ckpts/${DS_NAME}_finetune_${$EXP}/epoch6.pkl \
    --candidate_num 10 --example_num 100

# STAGE TWO
# Step 6
#bash scripts/prompt.sh \
#    --task $DS_NAME \
#    --version ${DS_NAME}_prompt_${$EXP} \
#    --examples_path outputs/results/${DS_NAME}_heuristics_${$EXP}/examples.json \
#    --candidates_path outputs/results/${DS_NAME}_heuristics_${$EXP}/candidates.json \
