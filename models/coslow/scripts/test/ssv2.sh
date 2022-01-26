#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coslow
DATASET=ssv2
GPUS=2
DISTRIBUTED_BACKEND=ddp
PRECISION=16

# Run test sequence ######################

python $PROJECT/main.py \
    --id CoSlow_ssv2_clip \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_ssv2.yaml \
    --batch_size 32 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode clip \
    --finetune_from_weights models/coslow/weights/slow_8x8_ssv2.pyth \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \


python $PROJECT/main.py \
    --id CoSlow_ssv2_frames_8 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_ssv2.yaml \
    --finetune_from_weights models/coslow/weights/slow_8x8_ssv2.pyth \
    --batch_size 16 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode init_frame \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \
  

python $PROJECT/main.py \
    --id CoSlow_ssv2_frames_64 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_ssv2.yaml \
    --finetune_from_weights models/coslow/weights/slow_8x8_ssv2.pyth \
    --batch_size 8 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --temporal_window_size 64 \
    --co3d_forward_mode init_frame \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --test \
