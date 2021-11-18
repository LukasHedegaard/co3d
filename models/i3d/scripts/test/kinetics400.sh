#!/bin/bash

PROJECT=models/i3d
DATASET=kinetics400
GPUS="1"

python $PROJECT/main.py \
        --id "i3d_kinetics400" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 16 \
        --from_hparams_file $PROJECT/hparams/i3d.yaml \
        --finetune_from_weights $PROJECT/weights/I3D_8x8_R50.pkl \
        --test \
        --logging_backend wandb \
        --normalisation_style imagenet \
