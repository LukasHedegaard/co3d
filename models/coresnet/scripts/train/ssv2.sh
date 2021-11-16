#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coresnet
DATASET=ssv2
GPUS=7
DISTRIBUTED_BACKEND=ddp
PRECISION=16

python $PROJECT/main.py \
    --id CoSlow_ssv2_frames_8 \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coresnet/hparams/slow_8x8_ssv2.yaml \
    --finetune_from_weights models/coresnet/weights/slow_8x8_ssv2.pyth \
    --batch_size 4 \
    --benchmark True \
    --logging_backend tensorboard \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode init_frame \
    --learning_rate 0.01 \
    --train \
    --test \
    --max_epochs 1 \
    --distributed_backend $DISTRIBUTED_BACKEND \
    --checkpoint_every_n_steps 2000 \
  
