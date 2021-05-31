#!/bin/bash

PROJECT=models/cox3d
DATASET=kinetics400

GPUS="1" 
for MODEL in s m l
do
    echo "========================== $MODEL =========================="

    CUDA_VISIBLE_DEVICES=0 \
    python $PROJECT/main.py \
        --id CoX3D_$(echo $MODEL)_profile_kinetics400 \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 64 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --profile_model \
        --log_level INFO \
        --logging_backend tensorboard \
        --co3d_forward_mode frame \
        --co3d_temporal_fill zeros \

    echo "========================== $MODEL $FRAMES_PER_CLIP =========================="

    CUDA_VISIBLE_DEVICES=0 \
    python $PROJECT/main.py \
        --id CoX3D_$(echo $MODEL)_64_profile_kinetics400 \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --profile_model \
        --log_level INFO \
        --logging_backend tensorboard \
        --co3d_forward_mode frame \
        --frames_per_clip 32 \
        --co3d_temporal_fill zeros \

done
