#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

MODEL="slow"
DATASET="kinetics400"
BATCH_SIZE=8
logging_backend="wandb"
GPUS=1

python models/slow/main.py \
    --id "${MODEL}_profile_${DATASET}" \
    --from_hparams_file $PROJECT/hparams/slow_8x8.yaml \
    --dataset $DATASET \
    --seed 123 \
    --profile_model \
    --gpus $GPUS \
    --batch_size $BATCH_SIZE \
    --logging_backend $LOGGING_BACKEND \

