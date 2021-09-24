#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=thumos14
GPUS=3
DISTRIBUTED_BACKEND=ddp
PRECISION=32

# Run test sequence ######################
# Allow for repetition of last frame up to 70% of steady-state (heuristic choice)


MODEL=s
FRAMES_PER_CLIP=13
FORWARD_FRAME_DELAY=$(($FRAMES_PER_CLIP + 56))
NUM_FORWARD_FRAMES=$((128 - $FORWARD_FRAME_DELAY))
BATCH_SIZE=16
LR=0.05 # $BATCH_SIZE * $GPUS * $NUM_FORWARD_FRAMES * OLD_LR / OLD_BS == 21

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames" \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size $BATCH_SIZE \
    --benchmark True \
    --train \
    --max_epochs 30 \
    --logging_backend wandb \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --frames_per_clip $FRAMES_PER_CLIP \
    --precision $PRECISION \
    --optimization_metric mAP \
    --unfreeze_from_epoch 0 \
    --learning_rate $LR \
    --num_sanity_val_steps 0 \
    --unfreeze_layer_step 2 \
    --distributed_backend $DISTRIBUTED_BACKEND \
