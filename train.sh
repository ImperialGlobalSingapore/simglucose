#!/bin/bash

# Login to Weights & Biases (if needed)
wandb login   # Replace with your actual W&B API key

# Set W&B environment variables
export WANDB_PROJECT="llm-glucose"  # Replace with your W&B project name
export WANDB_NAME="Qwen2-0.5B-SFT-Run"    # Name for this specific run
export WANDB_WATCH="all"  # Log gradients and parameters
export WANDB_LOG_MODEL="end"

# Run training with Accelerate
# accelerate launch --config_file default_config.yaml train_llama.py \

CUDA_VISIBLE_DEVICES=0 python train_llama.py \
    --model_name_or_path="Qwen/Qwen2-0.5B-Instruct" \
    --dataset_name="/data/shared-cache/glucose_questions_answers.jsonl" \
    --learning_rate=2.0e-5 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --logging_steps=25 \
    --eval_strategy=epoch \
    #--eval_steps=1000 \
    --output_dir="Qwen2-0.5B-SFT"

