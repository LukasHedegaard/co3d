#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=thumos14
GPUS=1
PRECISION=16

## Run train sequence

MODEL=s
FRAMES_PER_CLIP=5
FORWARD_FRAME_DELAY=64
NUM_FORWARD_FRAMES=128
LR=0.3 # $BATCH_SIZE * $GPUS * $NUM_FORWARD_FRAMES * OLD_LR / OLD_BS == 21

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames_weighted_ce" \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --accumulate_grad_batches 2 \
    --train \
    --max_epochs 20 \
    --logging_backend wandb \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --precision $PRECISION \
    --optimization_metric mAP \
    --learning_rate $LR \
    --num_sanity_val_steps 0 \
    --x3d_dropout_rate 0.37 \
    --weight_decay 0.0006 \
    --distributed_backend ddp \
    --unfreeze_from_epoch 0 \
    --unfreeze_epoch_step 20 \
    --mean_average_precision_skip_classes 0,21 \
    --log_every_n_steps 1 \
    --dataloader_prefetch_factor 4 \



# MODEL=l
# FRAMES_PER_CLIP=64
# FORWARD_FRAME_DELAY=64
# NUM_FORWARD_FRAMES=64
# LR=0.3 # $BATCH_SIZE * $GPUS * $NUM_FORWARD_FRAMES * OLD_LR / OLD_BS == 21

# python $PROJECT/main.py \
#     --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames" \
#     --dataset $DATASET \
#     --seed 42 \
#     --gpus $GPUS \
#     --from_hparams_file models/x3d/hparams/$MODEL.yaml \
#     --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
#     --batch_size 4 \
#     --accumulate_grad_batches 4 \
#     --train \
#     --max_epochs 20 \
#     --logging_backend wandb \
#     --num_workers 8 \
#     --co3d_forward_mode init_frame \
#     --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
#     --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
#     --step_between_clips $NUM_FORWARD_FRAMES \
#     --temporal_window_size $TEMPORAL_WINDOW_SIZE \
#     --precision $PRECISION \
#     --optimization_metric mAP \
#     --learning_rate $LR \
#     --num_sanity_val_steps 0 \
#     --x3d_dropout_rate 0.6 \
#     --weight_decay 0.0006 \
#     --distributed_backend ddp \
#     --unfreeze_from_epoch 0 \
#     --unfreeze_epoch_step 100 \
#     --mean_average_precision_skip_classes 0,21 \
#     --log_every_n_steps 1 \
#     --dataloader_prefetch_factor 4 \
#     --rand_augment_magnitude 15 \
#     --rand_augment_num_layers 4 \
