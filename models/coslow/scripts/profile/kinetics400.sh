#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi


python models/coslow/main.py \
    --id CoSlow_profile_kinetics400 \
    --dataset kinetics400 \
    --gpus 1 \
    --seed 123 \
    --batch_size 8 \
    --from_hparams_file models/coslow/hparams/slow_8x8_kinetics.yaml \
    --profile_model \
    --logging_backend wandb \
    --co3d_forward_mode frame \


python models/coslow/main.py \
    --id CoSlow_64_profile_kinetics400 \
    --dataset kinetics400 \
    --gpus 1 \
    --seed 123 \
    --batch_size 8 \
    --from_hparams_file models/coslow/hparams/slow_8x8_kinetics.yaml \
    --profile_model \
    --logging_backend wandb \
    --co3d_forward_mode frame \
    --temporal_window_size 64 \
