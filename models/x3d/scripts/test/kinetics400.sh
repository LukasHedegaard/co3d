#!/bin/bash

PROJECT=models/x3d
DATASET=kinetics400
GPUS="1"

for MODEL in xs s m l
do

    python $PROJECT/main.py \
        --id "x3d_{$MODEL}_kinetics400" \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 32 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
        --log_level DEBUG \
        --test \
        --test_ensemble 0 \

    # python $PROJECT/main.py \
    #     --id "x3d_{$MODEL}_kinetics400" \
    #     --results_log_dir $PROJECT \
    #     --dataset $DATASET \
    #     --gpus $GPUS \
    #     --seed 123 \
    #     --batch_size 32 \
    #     --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    #     --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    #     --log_level DEBUG \
    #     --test \
    #     --test_ensemble 1 \
    #     --test_ensemble_temporal_clips 10 \
    #     --test_ensemble_spatial_sampling_strategy "center" \

    # python $PROJECT/main.py \
    #     --id "x3d_{$MODEL}_kinetics400" \
    #     --results_log_dir $PROJECT \
    #     --dataset $DATASET \
    #     --gpus $GPUS \
    #     --seed 123 \
    #     --batch_size 32 \
    #     --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    #     --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    #     --log_level DEBUG \
    #     --test \
    #     --test_ensemble 1 \
    #     --test_ensemble_temporal_clips 10 \
    #     --test_ensemble_spatial_sampling_strategy "horizontal" \

done