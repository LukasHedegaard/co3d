#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=charades
GPUS=2
DISTRIBUTED_BACKEND=ddp
PRECISION=32

# Run test sequence ######################

python $PROJECT/main.py \
    --id CoSlow_charades_clip \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_charades.yaml \
    --batch_size 32 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode clip \
    --finetune_from_weights models/coresnet/weights/slow_8x8_charades.pyth \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \


python $PROJECT/main.py \
    --id CoSlow_charades_frames_8 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_charades.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_charades.pyth \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode frame \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \
  

python $PROJECT/main.py \
    --id CoSlow_charades_frames_64 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_charades.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_charades.pyth \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --frames_per_clip 64 \
    --co3d_forward_mode frame \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \
