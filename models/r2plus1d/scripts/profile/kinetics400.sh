#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

MODEL="r2plus1d"
DATASET="kinetics400"
DEVICE="RTX2080Ti"
GPUS=1
LOGGING_BACKEND="wandb"


python models/r2plus1d/main.py \
    --id "${MODEL}_8_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --seed 123 \
    --batch_size 32 \
    --profile_model \
    --gpus $GPUS \
    --temporal_window_size 8 \
    --logging_backend $LOGGING_BACKEND \


python models/r2plus1d/main.py \
    --id "${MODEL}_16_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --seed 123 \
    --batch_size 32 \
    --profile_model \
    --gpus $GPUS \
    --temporal_window_size 16 \
    --logging_backend $LOGGING_BACKEND \
