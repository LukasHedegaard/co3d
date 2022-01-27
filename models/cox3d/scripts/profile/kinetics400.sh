#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi


MODEL="cox3d"
DATASET=kinetics400
BATCH_SIZE=64
GPUS=1 
LOGGING_BACKEND="wandb"

for SIZE in s m
do
    echo "========================== ${SIZE} =========================="

    python models/cox3d/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \
        --co3d_forward_mode frame \

    echo "========================== ${SIZE} 64 =========================="

    python $PROJECT/main.py \
        --id "${MODEL}_${SIZE}_64_profile_${DATASET}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \
        --co3d_forward_mode frame \
        --temporal_window_size 64 \

done

SIZE=l
BATCH_SIZE=32

echo "========================== ${SIZE} =========================="

python models/cox3d/main.py \
    --id "${MODEL}_${SIZE}_profile_${DATASET}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \

echo "========================== ${SIZE} 64 =========================="

python $PROJECT/main.py \
    --id "${MODEL}_${SIZE}_64_profile_${DATASET}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --temporal_window_size 64 \