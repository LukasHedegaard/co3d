#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=ava
GPUS=1
PRECISION=16

# Run test sequence ######################

python $PROJECT/main.py \
    --id CoSlow_ava_clip \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_4x16_ava.yaml \
    --batch_size 1 \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode clip \
    --finetune_from_weights models/coresnet/weights/slow_4x16_ava.pyth \
    --test \


python $PROJECT/main.py \
    --id CoSlow_ava_frames_4 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_4x16_ava.yaml \
    --batch_size 1 \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode init_frame \
    --finetune_from_weights models/coresnet/weights/slow_4x16_ava.pyth \
    --test \


python $PROJECT/main.py \
    --id CoSlow_ava_frames_64 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_4x16_ava.yaml \
    --batch_size 1 \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode init_frame \
    --finetune_from_weights models/coresnet/weights/slow_4x16_ava.pyth \
    --test \
    --temporal_window_size 64 \
