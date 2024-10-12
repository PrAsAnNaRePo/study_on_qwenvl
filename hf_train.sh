#!/bin/bash

NUM_GPUS=2  # Set this to the number of GPUs you have
BATCH_SIZE=1  # Adjust based on your GPU memory
OUTPUT_DIR=./output
NUM_EPOCHS=7
DATA_PATH=./data.json
MAX_LEN=1024
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

deepspeed --num_gpus=$NUM_GPUS hftrain.py \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --data_path $DATA_PATH \
    --max_len $MAX_LEN \
    --model_id $MODEL_ID \
    --deepspeed zero3.json \
    --fp16 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5
