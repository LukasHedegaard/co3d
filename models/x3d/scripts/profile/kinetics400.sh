#!/bin/bash

PROJECT=models/co3d
DATASET=kinetics400micro
GPUS=1

for MODEL in s m
do

    CUDA_VISIBLE_DEVICES=4 python $PROJECT/main.py \
        --id x3d_profile_kinetics400 \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 64 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --profile_model \
        --log_level DEBUG \
        --precision 16 \

done

CUDA_VISIBLE_DEVICES=4 python $PROJECT/main.py \
        --id x3d_profile_kinetics400 \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 32 \
        --from_hparams_file $PROJECT/hparams/l.yaml \
        --profile_model \
        --log_level DEBUG \
        --precision 16 \
