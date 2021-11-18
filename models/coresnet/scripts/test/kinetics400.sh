#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=kinetics400
GPUS=1
PRECISION=16

# Run test sequence ######################

python $PROJECT/main.py \
    --id CoSlow_kin_clip \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --batch_size 8 \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode clip \


python $PROJECT/main.py \
    --id CoSlow_kin_frames_26 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --batch_size 8 \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode init_frame \
  

python $PROJECT/main.py \
    --id CoSlow_kin_frames_64 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_kinetics.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_kinetics.pyth \
    --batch_size 8 \
    --benchmark True \
    --test \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --temporal_window_size 64 \
    --co3d_forward_mode init_frame \

