#!/bin/bash

PROJECT=models/x3d
DATASET=kinetics400
GPUS="0," #1,2,3"

python $PROJECT/main.py \
    --id x3d_s_kinetics_finetune \
    --results_log_dir $PROJECT \
    --dataset $DATASET \
    --gpus $GPUS \
    --max_epochs 2 \
    --seed 123 \
    --optimization_metric top1acc \
    --learning_rate 0.01 \
    --discriminative_lr_fraction 0.05 \
    --batch_size 64 \
    --from_hparams_file $PROJECT/hparams/s.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_s.pyth \
    --unfreeze_from_epoch 0 \
    --unfreeze_layers_initial -1 \
    --train \
    --test \
    --notify \
    --log_level DEBUG \
    # --distributed_backend ddp \

