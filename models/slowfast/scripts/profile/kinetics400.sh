#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

MODEL="slowfast"
DATASET="kinetics400"
BATCH_SIZE=16
logging_backend="wandb"
GPUS=1


for SIZE in 4x16_R50 8x8_R50
do
    python models/slowfast/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}" \
        --dataset $DATASET \
        --seed 123 \
        --gpus $GPUS \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/slowfast/hparams/$SIZE.yaml \
        --logging_backend $LOGGING_BACKEND \
        --profile_model \
        --image_size 256 \

done
