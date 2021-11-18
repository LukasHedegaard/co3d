#!/bin/bash

# NB: For GPU-based devices, the largest bacth size possible will result in the highest throuhgput.

PROJECT=models/cox3d
DATASET=kinetics400
BATCH_SIZE=64

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
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --profile_model \
        --log_level INFO \
        --logging_backend tensorboard \
        --co3d_forward_mode frame \
        --co3d_temporal_fill zeros \

    echo "========================== $MODEL 64 =========================="

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
        --temporal_window_size 64 \
        --co3d_temporal_fill zeros \

done
