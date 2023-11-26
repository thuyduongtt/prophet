#!/bin/bash
# This script is used to prompt GPT-3 to generate final answers.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --examples_path)
      EXAMPLES_PATH="$2"
      shift 2;;
    --candidates_path)
      CANDIDATES_PATH="$2"
      shift 2;;
    --captions_path)
      CAPTIONS_PATH="$2"
      shift 2;;
#    --openai_key)
#      OPENAI_KEY="$2"
#      shift 2;;
    --llama_model)
      LLAMA_MODEL="$2"
      shift 2;;
    --llama_tokenizer)
      LLAMA_TOKENIZER="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
VERSION=${VERSION:-"prompt_okvqa"} # version name, default 'prompt_for_$TASK'
EXAMPLES_PATH=${EXAMPLES_PATH:-"assets/answer_aware_examples_okvqa.json"} # path to the examples, default is the result from our experiments
CANDIDATES_PATH=${CANDIDATES_PATH:-"assets/candidates_okvqa.json"} # path to the candidates, default is the result from our experiments
CAPTIONS_PATH=${CAPTIONS_PATH:-"assets/captions_okvqa.json"} # path to the captions, default is the result from our experiments
#OPENAI_KEY=${OPENAI_KEY:-""} # path to the captions

# CUDA_VISIBLE_DEVICES=$GPU \
#python main.py \
python -m torch.distributed.run --nproc_per_node 1 main.py \
    --task $TASK --run_mode prompt \
    --version $VERSION \
    --cfg configs/prompt_llama.yml \
    --examples_path $EXAMPLES_PATH \
    --candidates_path $CANDIDATES_PATH \
    --captions_path $CAPTIONS_PATH \
    --resume \
    --cache_version balanced_10_prompt_2_v2 \
    --prompt_file ${VERSION}.json \
    --export_prompt
