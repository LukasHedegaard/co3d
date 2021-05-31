#!/bin/bash

PROJECT=models/i3d
DATASET=kinetics400micro

# CPU
python $PROJECT/main.py \
    --id i3d_profile_kinetics400 \
    --from_hparams_file $PROJECT/hparams/i3d.yaml \
    --results_log_dir $PROJECT \
    --dataset $DATASET \
    --seed 123 \
    --batch_size 1 \
    --profile_model \
    --log_level DEBUG \
    --gpus 0 \

# GPU
# CUDA_VISIBLE_DEVICES=0 \
# python $PROJECT/main.py \
#     --id i3d_profile_kinetics400 \
#     --results_log_dir $PROJECT \
#     --dataset $DATASET \
#     --from_hparams_file $PROJECT/hparams/i3d.yaml \
#     --seed 123 \
#     --batch_size 16 \
#     --profile_model \
#     --log_level INFO \
#     --gpus 1 \
#     --precision 16 \

# XAVIER
# python $PROJECT/main.py \
#     --id i3d_profile_kinetics400 \
#     --results_log_dir $PROJECT \
#     --dataset $DATASET \
#     --from_hparams_file $PROJECT/hparams/i3d.yaml \
#     --seed 123 \
#     --batch_size 32 \
#     --profile_model \
#     --log_level INFO \
#     --gpus 1 \
#     --precision 16 \

