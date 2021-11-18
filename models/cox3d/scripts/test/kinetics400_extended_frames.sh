#!/bin/bash

PROJECT=models/cox3d
DATASET=kinetics400

## Run val sequence ######################

MAX_FRAME_DELAY=148 #10 sec x 30 fps / 2 steps - 2
FRAME_RATE=2

MODEL=s
FRAME_DELAY=56

for FRAMES_PER_CLIP in 96 64 32 16 13
do
    FORWARD_FRAME_DELAY=$(($FRAME_DELAY + $TEMPORAL_WINDOW_SIZE - 1))
    FORWARD_FRAME_DELAY=$(($FORWARD_FRAME_DELAY>$MAX_FRAME_DELAY ? $MAX_FRAME_DELAY : $FORWARD_FRAME_DELAY ))
    echo $FORWARD_FRAME_DELAY

    horovodrun -np 4 python $PROJECT/main.py \
        --id CoX3D_extended_frames_fr2_$MODEL \
        --dataset $DATASET \
        --seed 42 \
        --gpus 1 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
        --batch_size 32 \
        --co3d_forward_mode init_frame \
        --co3d_temporal_fill replicate \
        --co3d_num_forward_frames 1 \
        --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
        --benchmark True \
        --validate \
        --logging_backend wandb \
        --num_workers 5 \
        --temporal_window_size $TEMPORAL_WINDOW_SIZE \
        --precision 16 \
        --frame_rate $FRAME_RATE \

done


MODEL=m

for FRAMES_PER_CLIP in 96 64 32 16
do
    FORWARD_FRAME_DELAY=$(($FRAME_DELAY + $TEMPORAL_WINDOW_SIZE - 1))
    FORWARD_FRAME_DELAY=$(($FORWARD_FRAME_DELAY>$MAX_FRAME_DELAY ? $MAX_FRAME_DELAY : $FORWARD_FRAME_DELAY ))
    echo $FORWARD_FRAME_DELAY

    horovodrun -np 4 python $PROJECT/main.py \
        --id CoX3D_extended_frames_fr2_$MODEL \
        --dataset $DATASET \
        --seed 42 \
        --gpus 1 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
        --batch_size 16 \
        --co3d_forward_mode init_frame \
        --co3d_temporal_fill replicate \
        --co3d_num_forward_frames 1 \
        --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
        --benchmark True \
        --validate \
        --logging_backend wandb \
        --num_workers 5 \
        --temporal_window_size $TEMPORAL_WINDOW_SIZE \
        --precision 16 \
        --frame_rate $FRAME_RATE \

done


MODEL=l
FRAME_DELAY=114 # From conv layers in large model

for FRAMES_PER_CLIP in 16 #96 64 32
do
    FORWARD_FRAME_DELAY=$(($FRAME_DELAY + $TEMPORAL_WINDOW_SIZE - 1))
    FORWARD_FRAME_DELAY=$(($FORWARD_FRAME_DELAY>$MAX_FRAME_DELAY ? $MAX_FRAME_DELAY : $FORWARD_FRAME_DELAY ))
    echo $FORWARD_FRAME_DELAY

    horovodrun -np 4 python $PROJECT/main.py \
        --id CoX3D_extended_frames_fr2_$MODEL \
        --dataset $DATASET \
        --seed 42 \
        --gpus 1 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
        --batch_size 8 \
        --co3d_forward_mode init_frame \
        --co3d_temporal_fill replicate \
        --co3d_num_forward_frames 1 \
        --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
        --benchmark True \
        --validate \
        --logging_backend wandb \
        --num_workers 5 \
        --temporal_window_size $TEMPORAL_WINDOW_SIZE \
        --frame_rate $FRAME_RATE \
        # --precision 16 \

done



# Run test sequence ######################
# Rounding number of frames down to limit to clip size

MODEL=s
MAX_FRAME_DELAY=48 #10 sec x 30 fps / 6 steps - 2
FRAME_DELAY=56

for FRAMES_PER_CLIP in 64 32 16 13
do
    for FRAMES_PER_CLIP in 64 32 16
    do
        horovodrun -np 4 python $PROJECT/main.py \
            --id CoX3D_extended_frames_fr2_$MODEL \
            --dataset $DATASET \
            --seed 42 \
            --gpus 1 \
            --from_hparams_file models/x3d/hparams/$MODEL.yaml \
            --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
            --batch_size 16 \
            --co3d_forward_mode init_frame \
            --co3d_temporal_fill replicate \
            --co3d_num_forward_frames 1 \
            --co3d_forward_frame_delay $(($FRAME_DELAY + $TEMPORAL_WINDOW_SIZE - 1))  \
            --benchmark True \
            --validate \
            --distributed_backend horovod \
            --logging_backend wandb \
            --num_workers 6 \
            --temporal_window_size $TEMPORAL_WINDOW_SIZE \
            --precision 16 \
            --temporal_downsampling 2 \

    done  
done


MODEL=m

# Use max avilable num frames
horovodrun -np 4 python $PROJECT/main.py \
            --id CoX3D_extended_frames_fr2_$MODEL \
            --dataset $DATASET \
            --seed 42 \
            --gpus 1 \
            --from_hparams_file models/x3d/hparams/$MODEL.yaml \
            --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
            --batch_size 8 \
            --co3d_forward_mode init_frame \
            --co3d_temporal_fill replicate \
            --co3d_num_forward_frames 1 \
            --co3d_forward_frame_delay 148  \
            --benchmark True \
            --validate \
            --distributed_backend horovod \
            --logging_backend wandb \
            --num_workers 6 \
            --temporal_window_size 96 \
            --precision 16 \
            --temporal_downsampling 2 \

for FRAME_DELAY in 56
do
    for FRAMES_PER_CLIP in 64 32 16
    do
        horovodrun -np 4 python $PROJECT/main.py \
            --id CoX3D_extended_frames_fr2_$MODEL \
            --dataset $DATASET \
            --seed 42 \
            --gpus 1 \
            --from_hparams_file models/x3d/hparams/$MODEL.yaml \
            --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
            --batch_size 16 \
            --co3d_forward_mode init_frame \
            --co3d_temporal_fill replicate \
            --co3d_num_forward_frames 1 \
            --co3d_forward_frame_delay $(($FRAME_DELAY + $TEMPORAL_WINDOW_SIZE - 1))  \
            --benchmark True \
            --validate \
            --distributed_backend horovod \
            --logging_backend wandb \
            --num_workers 6 \
            --temporal_window_size $TEMPORAL_WINDOW_SIZE \
            --precision 16 \
            --temporal_downsampling 2 \

    done  
done


# Run test sequence ######################
Allow for repetition of last frame up to 70% of steady-state (heuristic choice)




MODEL=s
FRAMES_PER_CLIP=64
FORWARD_FRAME_DELAY=84 # (56 + 64) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \


MODEL=s
FRAMES_PER_CLIP=13
FORWARD_FRAME_DELAY=48 # (56+13) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \


MODEL=s
FRAMES_PER_CLIP=13
FORWARD_FRAME_DELAY=56

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \

MODEL=m
FRAMES_PER_CLIP=64
FORWARD_FRAME_DELAY=84 # (56 + 64) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \



MODEL=m
FRAMES_PER_CLIP=16
FORWARD_FRAME_DELAY=50 # (56+16) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \



MODEL=m
FRAMES_PER_CLIP=16
FORWARD_FRAME_DELAY=59

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \



MODEL=l
FRAMES_PER_CLIP=64
FORWARD_FRAME_DELAY=124 # (114 + 64) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 4 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 4 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \


MODEL=l
FRAMES_PER_CLIP=16
FORWARD_FRAME_DELAY=91 # (114 + 16) * 0.7

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 6 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \

MODEL=l
FRAMES_PER_CLIP=16
FORWARD_FRAME_DELAY=106 # (114 + 16) * 0.8

horovodrun -np 4 python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 6 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --validate \
    --distributed_backend horovod \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision 16 \