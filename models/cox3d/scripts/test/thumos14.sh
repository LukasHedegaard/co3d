#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=thumos14
GPUS=1
PRECISION=32

## Run train sequence

MODEL=s
FRAMES_PER_CLIP=5
FORWARD_FRAME_DELAY=64
NUM_FORWARD_FRAMES=264
LR=0.6 # $BATCH_SIZE * $GPUS * $NUM_FORWARD_FRAMES * OLD_LR / OLD_BS == 21

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames_extract_features" \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --batch_size 4 \
    --test \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --frames_per_clip $FRAMES_PER_CLIP \
    --precision $PRECISION \
    --optimization_metric mAP \
    --mean_average_precision_skip_classes 0,21 \
    --log_every_n_steps 1 \
    --finetune_from_weights /mnt/archive/lh/hags/logs/run_logs/CoX3DRide/178o5fsu/checkpoints/epoch=16-step=611.ckpt \
    --extract_features_after_layer module.head.projection \
    --limit_test_batches 10 \
    --logging_backend wandb \
    # --distributed_backend ddp \
    # --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    # --extract_features_after_layer module.head.projection \
    # --dataloader_prefetch_factor 4 \
    # --unfreeze_layer_step 2 \
    # --benchmark True \
    # --discriminative_lr_fraction 0.001 \

