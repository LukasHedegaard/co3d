#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi


MODEL="x3d"
DATASET=kinetics400
BATCH_SIZE=64
GPUS=1 
LOGGING_BACKEND="wandb"

for SIZE in xs s m
do
    python models/x3d/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \


done

SIZE=l
BATCH_SIZE=32

python models/x3d/main.py \
    --id "${MODEL}_${SIZE}_profile_${DATASET}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
