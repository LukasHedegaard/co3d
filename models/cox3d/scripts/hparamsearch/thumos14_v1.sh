#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=thumos14
GPUS=1
PRECISION=16

## Small model

MODEL=s
FRAMES_PER_CLIP=1
FORWARD_FRAME_DELAY=64
NUM_FORWARD_FRAMES=64
BATCH_SIZE=2

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_hparamsearch" \
    --dataset $DATASET \
    --seed 42 \
    --from_hparams_file /home/lh/projects/co3d/models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights /home/lh/projects/co3d/models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size $BATCH_SIZE \
    --gpus 4 \
    --hparamsearch \
    --gpus_per_trial 1 \
    --trials 30 \
    --max_epochs 20 \
    --logging_backend wandb \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --precision $PRECISION \
    --optimization_metric mAP \
    --num_sanity_val_steps 0 \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --from_hparam_space_file models/cox3d/scripts/hparamsearch/s_space.yaml \
    --unfreeze_epoch_step 3 \
    --unfreeze_layer_step -1 \
    --unfreeze_from_epoch 0 \
    --accumulate_grad_batches 8 \
    # --limit_train_batches 2 \
    # --limit_val_batches 2 \

