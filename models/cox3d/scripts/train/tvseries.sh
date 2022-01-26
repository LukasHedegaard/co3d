#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=tvseries
GPUS=1
PRECISION=32

## Run train sequence

MODEL=s
FRAMES_PER_CLIP=13
NUM_FORWARD_FRAMES=8
LR=0.1

python $PROJECT/main.py \
    --id "CoX3D_${MODEL}_${DATASET}_${FRAMES_PER_CLIP}_frames" \
    --dataset $DATASET \
    --metric_selection $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --batch_size 16 \
    --train \
    --max_epochs 30 \
    --logging_backend wandb \
    --num_workers 8 \
    --co3d_forward_mode init_frame \
    --co3d_num_forward_frames $NUM_FORWARD_FRAMES \
    --step_between_clips $NUM_FORWARD_FRAMES \
    --frames_per_clip $FRAMES_PER_CLIP \
    --precision $PRECISION \
    --optimization_metric mcAP \
    --learning_rate $LR \
    --num_sanity_val_steps 0 \
    --x3d_dropout_rate 0.5 \
    --weight_decay 0.0006 \
    --unfreeze_from_epoch 0 \
    --unfreeze_layer_step 1 \
    --unfreeze_epoch_step 100 \
    --mean_average_precision_skip_classes 0 \
    --log_every_n_steps 1 \
    --rand_augment_magnitude 11 \
    --rand_augment_num_layers 4 \

