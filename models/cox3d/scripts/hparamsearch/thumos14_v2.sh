#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=thumos14
PRECISION=16

## Small model

MODEL=s
FRAMES_PER_CLIP=64
FORWARD_FRAME_DELAY=64
NUM_FORWARD_FRAMES=64

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames_hparamsearch" \
    --dataset $DATASET \
    --seed 42 \
    --from_hparams_file /home/lh/projects/co3d/models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights /mnt/archive/lh/hags/logs/run_logs/CoX3DRide/2e5ndjl5/checkpoints/epoch=18-step=455.ckpt \
    --batch_size 2 \
    --accumulate_grad_batches 8 \
    --gpus 5 \
    --hparamsearch \
    --gpus_per_trial 1 \
    --trials 30 \
    --max_epochs 20 \
    --logging_backend wandb \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --frames_per_clip $FRAMES_PER_CLIP \
    --precision $PRECISION \
    --optimization_metric mAP \
    --num_sanity_val_steps 0 \
    --from_hparam_space_file models/cox3d/scripts/hparamsearch/s_space.yaml \
    --mean_average_precision_skip_classes 0,21 \
    --dataloader_prefetch_factor 4 \
    --log_every_n_steps 1 \
    --unfreeze_epoch_step 1 \
    --unfreeze_layer_step 5 \
    --unfreeze_from_epoch 0 \
    --unfreeze_layers_initial 5 \

