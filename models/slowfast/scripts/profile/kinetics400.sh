#!/bin/bash

PROJECT=models/slowfast
DATASET=kinetics400

# CPU
for MODEL in 4x16_R50 8x8_R50
do
    python $PROJECT/main.py \
        --id slowfast_profile_kinetics400 \
        --dataset $DATASET \
        --seed 123 \
        --batch_size 16 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --profile_model \
        --gpus 0 \
        --image_size 256 \

done

# GPU
# for MODEL in 4x16_R50 8x8_R50
# do
#     CUDA_VISIBLE_DEVICES=0 \
#     python $PROJECT/main.py \
#         --id slowfast_profile_kinetics400 \
#         --dataset $DATASET \
#         --seed 123 \
#         --batch_size 16 \
#         --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
#         --profile_model \
#         --gpus 1 \
#         --image_size 256 \

# done

# XAVIER
# MODEL in 4x16_R50 8x8_R50
# do
#     python $PROJECT/main.py \
#         --id slowfast_profile_kinetics400 \
#         --dataset $DATASET \
#         --seed 123 \
#         --batch_size 32 \
#         --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
#         --profile_model \
#         --gpus 1 \
#         --image_size 256 \
#         --precision 16 \

# done
