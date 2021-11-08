#!/bin/bash

PROJECT=models/r2plus1d
DATASET=kinetics400

# CPU
# python $PROJECT/main.py \
#     --id r2plus1d_profile_kinetics400 \
#     --results_log_dir $PROJECT \
#     --dataset $DATASET \
#     --seed 123 \
#     --batch_size 1 \
#     --profile_model \
#     --log_level DEBUG \
#     --gpus 0 \
#     --temporal_window_size 64 \


# GPU
# CUDA_VISIBLE_DEVICES=0 \
# python $PROJECT/main.py \
#     --id r2plus1d_profile_kinetics400 \
#     --results_log_dir $PROJECT \
#     --dataset $DATASET \
#     --seed 123 \
#     --batch_size 16 \
#     --profile_model \
#     --log_level INFO \
#     --gpus 1 \
#     --temporal_window_size 8 \

# XAVIER
python $PROJECT/main.py \
    --id r2plus1d_profile_kinetics400 \
    --results_log_dir $PROJECT \
    --dataset $DATASET \
    --from_hparams_file $PROJECT/hparams/i3d.yaml \
    --seed 123 \
    --batch_size 32 \
    --profile_model \
    --log_level INFO \
    --gpus 1 \
    --precision 16 \
