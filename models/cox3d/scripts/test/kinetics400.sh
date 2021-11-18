#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=kinetics400
GPUS=3
DISTRIBUTED_BACKEND=ddp
PRECISION=32

# Run test sequence ######################
# Allow for repetition of last frame up to 70% of steady-state (heuristic choice)


MODEL=l
TEMPORAL_WINDOW_SIZE=64
FORWARD_FRAME_DELAY=124 # (114 + 64) * 0.7

python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 4 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 4 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision $PRECISION \
    --distributed_backend $DISTRIBUTED_BACKEND \


MODEL=m
TEMPORAL_WINDOW_SIZE=64
FORWARD_FRAME_DELAY=84 # (56 + 64) * 0.7

python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 8 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --limit_test_batches 30 \
    --precision $PRECISION \
    --distributed_backend $DISTRIBUTED_BACKEND \


MODEL=s
TEMPORAL_WINDOW_SIZE=64
FORWARD_FRAME_DELAY=84 # (56 + 64) * 0.7

python $PROJECT/main.py \
    --id CoX3D_extended_frames_test_pad_$MODEL \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 5 \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --co3d_num_forward_frames 1 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --precision $PRECISION \
    --distributed_backend $DISTRIBUTED_BACKEND \

