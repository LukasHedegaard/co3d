#!/bin/bash

PROJECT=models/cox3d
DATASET=kinetics400


for MODEL in s m
do
    # Replicate
    for FRAME_DELAY in 0 16 24 26 28 30 32 36 40 44 48 52 56 60 64 68 72 80
    do
        horovodrun -np 4 python $PROJECT/main.py \
            --id CoX3D_transient_replicate_fr2_$MODEL \
            --dataset $DATASET \
            --seed 42 \
            --gpus 1 \
            --from_hparams_file models/x3d/hparams/$MODEL.yaml \
            --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
            --batch_size 16 \
            --log_level INFO \
            --co3d_forward_mode init_frame \
            --co3d_temporal_fill replicate \
            --co3d_num_forward_frames 1 \
            --co3d_forward_frame_delay $FRAME_DELAY \
            --benchmark True \
            --validate \
            --distributed_backend horovod \
            --logging_backend wandb \
            --num_workers 6 \
            --precision 16 \
            --temporal_downsampling 2 \
            
    done

    # Zeros
    for FRAME_DELAY in 0 16 24 26 28 30 32 36 38 40 42 44 48 52 56 60 64 68 72
    do
        horovodrun -np 4 python $PROJECT/main.py \
            --id CoX3D_transient_zeros_fr2_$MODEL \
            --dataset $DATASET \
            --seed 42 \
            --gpus 1 \
            --from_hparams_file models/x3d/hparams/$MODEL.yaml \
            --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
            --batch_size 32 \
            --log_level INFO \
            --co3d_forward_mode init_frame \
            --co3d_temporal_fill zeros \
            --co3d_num_forward_frames 1 \
            --co3d_forward_frame_delay $FRAME_DELAY \
            --benchmark True \
            --validate \
            --distributed_backend horovod \
            --logging_backend wandb \
            --num_workers 6 \
            --precision 16 \
            --temporal_downsampling 2 \
            
    done
done