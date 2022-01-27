#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi


python models/i3d/main.py \
    --id i3d_profile_kinetics400 \
    --dataset kinetics400 \
    --gpus 1 \
    --seed 123 \
    --batch_size 8 \
    --from_hparams_file models/i3d/hparams/i3d.yaml \
    --profile_model \
    --logging_backend wandb \
