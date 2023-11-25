#!/bin/bash

N=2
VERSION=2

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
#bash scripts/pretrain.sh --task $DS_NAME --version ${DS_NAME}_pretrain_${N}

# Step 4
# bash scripts/finetune.sh --task $DS_NAME --version ${DS_NAME}_finetune_${N} --pretrained_model ckpts/mcan_pt_okvqa.pkl
#bash scripts/finetune.sh \
# --task $DS_NAME \
# --version ${DS_NAME}_finetune_${N} \
# --pretrained_model outputs/ckpts/${DS_NAME}_pretrain_${N}/epoch13.pkl

# Step 5
#bash scripts/heuristics_gen.sh \
#    --task $DS_NAME \
#    --version ${DS_NAME}_heuristics_${N} \
#    --ckpt_path outputs/ckpts/${DS_NAME}_finetune_${N}/epoch6.pkl \
#    --candidate_num 10 --example_num 100

# STAGE TWO
# Step 6
source ~/.profile  # make sure the OpenAI Key is loaded
bash scripts/prompt.sh \
    --task $DS_NAME \
    --version ${DS_NAME}_prompt_${N}_${VERSION} \
    --examples_path outputs/results/${DS_NAME}_heuristics_${N}/examples.json \
    --candidates_path outputs/results/${DS_NAME}_heuristics_${N}/candidates.json \
    --captions_path assets/captions_${DS_NAME}.json \
    --llama_model ../Llama/llama-2-13b-chat/ \
    --llama_tokenizer ../Llama/tokenizer.model
