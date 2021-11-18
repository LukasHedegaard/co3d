#!/bin/bash

PROJECT=models/r2plus1d
DATASET=kinetics400

# CPU
python $PROJECT/main.py \
    --id r2plus1d_profile_kinetics400 \
    --dataset $DATASET \
    --seed 123 \
    --batch_size 1 \
    --profile_model \
    --gpus 0 \
    --temporal_window_size 8 \

# GPU
# CUDA_VISIBLE_DEVICES=0 \
# python $PROJECT/main.py \
#     --id r2plus1d_profile_kinetics400 \
#     --dataset $DATASET \
#     --seed 123 \
#     --batch_size 16 \
#     --profile_model \
#     --gpus 1 \
#     --temporal_window_size 8 \

# XAVIER
# python $PROJECT/main.py \
#     --id r2plus1d_profile_kinetics400 \
#     --dataset $DATASET \
#     --from_hparams_file $PROJECT/hparams/i3d.yaml \
#     --seed 123 \
#     --batch_size 32 \
#     --profile_model \
#     --temporal_window_size 8 \
#     --gpus 1 \
#     --precision 16 \
