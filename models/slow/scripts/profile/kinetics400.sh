#!/bin/bash

PROJECT=models/slow
DATASET=kinetics400

# CPU
python $PROJECT/main.py \
    --id slow_profile_kinetics400 \
    --from_hparams_file $PROJECT/hparams/slow.yaml \
    --dataset $DATASET \
    --seed 123 \
    --batch_size 1 \
    --profile_model \
    --gpus 0 \

# GPU
# CUDA_VISIBLE_DEVICES=0 \
# python $PROJECT/main.py \
#     --id slow_profile_kinetics400 \
#     --dataset $DATASET \
#     --from_hparams_file $PROJECT/hparams/slow.yaml \
#     --seed 123 \
#     --batch_size 16 \
#     --profile_model \
#     --gpus 1 \
#     --precision 16 \

# XAVIER
# python $PROJECT/main.py \
#     --id slow_profile_kinetics400 \
#     --dataset $DATASET \
#     --from_hparams_file $PROJECT/hparams/slow.yaml \
#     --seed 123 \
#     --batch_size 32 \
#     --profile_model \
#     --gpus 1 \
#     --precision 16 \

