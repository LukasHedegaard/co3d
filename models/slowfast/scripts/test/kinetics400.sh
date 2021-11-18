#!/bin/bash

PROJECT=models/slowfast
DATASET=kinetics400
GPUS="1"

for MODEL in 4x16_R50 8x8_R50
do

    python $PROJECT/main.py \
        --id slowfast_test_kinetics400 \
        --dataset $DATASET \
        --seed 123 \
        --batch_size 16 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --finetune_from_weights $PROJECT/weights/SLOWFAST_$MODEL.pkl \
        --test \
        --gpus $GPUS \
        --logging_backend wandb \
        --image_size 256 \

done