#!/bin/bash

# config
OUTPUT_DIR="models/bge-m3-finetuned"
TRAIN_DATA="data/"
NUM_GPUS=1 
BATCH_SIZE=1 
NUM_EPOCHS=1

mkdir -p $OUTPUT_DIR
torchrun --nproc_per_node $NUM_GPUS \
    -m FlagEmbedding.BGE_M3.run \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path BAAI/bge-m3 \
    --train_data $TRAIN_DATA \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 512 \
    --passage_max_len 8000 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --same_task_within_batch True \
    --unified_finetuning True \
    --use_self_distill True


# pip install transformers==4.45.2
# pip install -U FlagEmbedding
# pip install peft
# pip install sentencepiece