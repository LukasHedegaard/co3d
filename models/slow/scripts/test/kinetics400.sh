#!/bin/bash

PROJECT=models/slow

python $PROJECT/main.py \
        --id "slow_kinetics400" \
        --dataset kinetics400 \
        --gpus 1 \
        --seed 123 \
        --batch_size 16 \
        --from_hparams_file $PROJECT/hparams/slow_8x8.yaml \
        --finetune_from_weights $PROJECT/weights/slow_8x8_kinetics.pyth \
        --test \
        --logging_backend wandb \
