#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=kinetics400
GPUS=1
DISTRIBUTED_BACKEND=ddp
PRECISION=32

# Run test sequence ######################
# Allow for repetition of last frame up to 70% of steady-state (heuristic choice)

FRAMES_PER_CLIP=64
FORWARD_FRAME_DELAY=124 # (114 + 64) * 0.7

python $PROJECT/main.py \
    --id CoSlow_kin \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --batch_size 4 \
    --co3d_forward_mode init_frame \
    --co3d_temporal_fill replicate \
    --co3d_num_forward_frames 1 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 4 \
    --frames_per_clip $FRAMES_PER_CLIP \
    --precision $PRECISION \
    --distributed_backend $DISTRIBUTED_BACKEND \
