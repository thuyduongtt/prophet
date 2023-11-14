#!/bin/bash

DS_NAME=unbalanced

# ==============================
# ===== DATA PREPARATION
# 1. Download the dataset (unbalanced and balanced_10) and place into "datasets" folder
# Better do that manually

# 2. Extract image features by running
bash scripts/extract_img_feats.sh --dataset $DS_NAME


# ==============================
# ===== STAGE ONE
# 3-1. Download the pre-trained model (on OK-VQA)
# wget https://awma1-my.sharepoint.com/personal/yuz_l0_tn/Documents/share/prophet/ckpts/mcan_pt_okvqa.pkl -P ckpts/
cd ckpts/
FILE_ID='18MfupIR77iHd_vJKaSNGYCTCMTojKcxc'
gdown $ZIP_ID
cd ..

# 3-2. Pretrain - do we need this step?
bash scripts/pretrain.sh --version ${DS_NAME}_pretrain_1

# 4. Start finetuning on the target dataset
bash scripts/finetune.sh --version ${DS_NAME}_finetune_1 --pretrained_model ckpts/mcan_pt_okvqa.pkl
# All epoch checkpoints are saved in outputs/ckpts/{version}

# 5. Extract heuristic answers
bash scripts/heuristics_gen.sh \
    --version ${DS_NAME}_heuristics_1
    --ckpt_path outputs/ckpts/${DS_NAME}_finetune_1/epoch6.pkl
    --candidate_num 10 --example_num 100
# The extracted answer heuristics will be stored as candidates.json and examples.json in outputs/results/{version} directory.


# ==============================
# ===== STAGE TWO
# 6. Prompt GPT-3 with answer heuristics and generate better answers
$ bash scripts/prompt.sh \
    --version ${DS_NAME}_prompt_1 \
    --examples_path outputs/results/${DS_NAME}_heuristics_1/examples.json \
    --candidates_path outputs/results/${DS_NAME}_heuristics_1/candidates.json \
    --openai_key sk-xxxxxxxxxxxxxxxxxxxxxx
# The result file will be stored as result.json in outputs/results/{version} directory.


# ==============================
# ===== EVALUATION (optional)
$ bash scripts/evaluate_file.sh \
    --task ok --result_path outputs/results/${DS_NAME}_prompt_1/result.json
