#!/usr/bin/env bash

set -e

# CONFIG
TASK="modeling_comprehension" # modeling_comprehension, inference_capability, mistake_correction_competence, appropriate_vocabulary_usage
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# DATASET="../generation/data/2026_per_gen_expert_annotations_completePrompt_uniqueMistake.csv"
DATASET="../generation/data/2025_big_2026_small_annotated_data.csv"
OUTPUT_DIR="trained_adapters/"

EPOCHS=40
BATCH_SIZE=4
LR=5e-5
MAX_LENGTH=1024

LORA_R=4
LORA_ALPHA=16
LORA_DROPOUT=0.05

VAL_SPLIT=0.2
MISTAKE_TYPES_PATH="../prompts/modeling_mistakes.json"

TRAIN_EMBEDDINGS=true
TRAIN_EMBEDDINGS_FLAG=""
if [ "$TRAIN_EMBEDDINGS" = true ]; then
    TRAIN_EMBEDDINGS_FLAG="--train_embeddings"
fi

# RUN
echo "Starting training for task: $TASK"

python3 train_predictors.py \
  --task "$TASK" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --max_length $MAX_LENGTH \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --val_split $VAL_SPLIT \
  --mistake_types_path "$MISTAKE_TYPES_PATH" \
  $TRAIN_EMBEDDINGS_FLAG

echo "Training finished."