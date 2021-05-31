#!/bin/bash

PROJECT=models/r2plus1d
DATASET=kinetics400
GPUS="2,"

python $PROJECT/main.py \
        --id r2plus1d_kinetics400 \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 8 \
        --finetune_from_weights $PROJECT/weights/r2plus1d_18-91a641e6.pth \
        --log_level DEBUG \
        --test \
        --normalisation_style imagenet \
        --frames_per_clip 64 \
        --temporal_downsampling 1 \
