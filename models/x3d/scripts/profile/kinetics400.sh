#!/bin/bash

PROJECT=models/co3d
DATASET=kinetics400
GPUS=1

for MODEL in xs s m
do

    python $PROJECT/main.py \
        --id x3d_profile_kinetics400 \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size 64 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --profile_model \
        --precision 16 \

done

python $PROJECT/main.py \
    --id x3d_profile_kinetics400 \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size 32 \
    --from_hparams_file $PROJECT/hparams/l.yaml \
    --profile_model \
    --precision 16 \
