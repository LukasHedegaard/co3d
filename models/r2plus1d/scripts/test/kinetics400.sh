#!/bin/bash

PROJECT=models/r2plus1d
DATASET=kinetics400
GPUS="2,"

python $PROJECT/main.py \
        --id r2plus1d_kinetics400 \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 8 \
        --finetune_from_weights $PROJECT/weights/r2plus1d_18-91a641e6.pth \
        --test \
        --normalisation_style imagenet \
        --temporal_window_size 8 \
        --temporal_downsampling 1 \
