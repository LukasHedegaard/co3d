#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=kinetics400
PRECISION=32

# Run test sequence ######################


python $PROJECT/main.py \
    --id CoSlow_kinetics_frames_8 \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --co3d_forward_mode init_frame \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --test \


python $PROJECT/main.py \
    --id CoSlow_kinetics_frames_64 \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --co3d_forward_mode init_frame \
    --batch_size 1 \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --temporal_window_size 64 \
    --test \
