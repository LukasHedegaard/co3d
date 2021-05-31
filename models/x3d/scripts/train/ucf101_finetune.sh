#!/bin/bash

PROJECT=models/x3d
DATASET=ucf101
GPUS="0,"

python $PROJECT/main.py \
    --id x3d_xs \
    --results_log_dir $PROJECT \
    --dataset $DATASET \
    --gpus $GPUS \
    --max_epochs 20 \
    --seed 123 \
    --optimization_metric top1acc \
    --learning_rate 0.02 \
    --discriminative_lr_fraction 0.05 \
    --batch_size 128 \
    --from_hparams_file $PROJECT/hparams/xs.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_xs.pyth \
    --unfreeze_from_epoch 0 \
    --unfreeze_layers_initial 1 \
    --unfreeze_layer_step 20 \
    --unfreeze_epoch_step 2 \
    --unfreeze_layers_max 22 \
    --train \
    --test \
    --notify \
    --log_level DEBUG \

