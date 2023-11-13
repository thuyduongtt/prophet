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
bash scripts/extract_img_feats.sh --dataset $DS_NAME