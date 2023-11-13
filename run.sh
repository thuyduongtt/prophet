#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    ;;
  2)
    DS_NAME="balanced_10"
    ;;
esac


DS_DIR="../dataset/${DS_NAME}"
bash scripts/extract_img_feats.sh --dataset $DS_DIR