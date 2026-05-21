#!/usr/bin/env bash

set -e

# CONFIG
TASK="inference_capability" # modeling_comprehension, inference_capability, mistake_correction_competence, appropriate_vocabulary_usage
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# DATASET="../generation/data/2026_per_gen_expert_annotations_completePrompt_uniqueMistake.csv"
DATASET="../generation/data/2025_big_2026_small_annotated_data.csv"

BATCH_SIZE=4
MAX_LENGTH=1024

VAL_SPLIT=0.2
MISTAKE_TYPES_PATH="../prompts/modeling_mistakes.json"

# RUN
echo "Starting pretrained baseline evaluation for task: $TASK"

python3 pretrained_predictors.py \
  --task "$TASK" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --val_split $VAL_SPLIT \
  --mistake_types_path "$MISTAKE_TYPES_PATH"

echo "Evaluation finished."