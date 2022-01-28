#!/bin/bash

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi


DEVICE="Xavier"
DATASET="kinetics400"
BATCH_SIZE=8
LOGGING_BACKEND="wandb"
GPUS=1
PRECISION=16


MODEL="coi3d"

python3 models/coi3d/main.py \
    --id "${MODEL}_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/coi3d/hparams/i3d.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --precision $PRECISION \


python3 models/coi3d/main.py \
    --id "${MODEL}_64_profile_${DATASET}_${DEVICE}" \
    --dataset kinetics400 \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/coi3d/hparams/i3d.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --temporal_window_size 64 \
    --precision $PRECISION \




MODEL="coslow"

python3 models/coslow/main.py \
    --id "${MODEL}_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/coslow/hparams/slow_8x8_kinetics.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --precision $PRECISION \


python3 models/coslow/main.py \
    --id "${MODEL}_64_profile_${DATASET}_${DEVICE}" \
    --dataset kinetics400 \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/coslow/hparams/slow_8x8_kinetics.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --temporal_window_size 64 \
    --precision $PRECISION \




MODEL="cox3d"
BATCH_SIZE=64

for SIZE in s m
do
    python3 models/cox3d/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}_${DEVICE}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \
        --co3d_forward_mode frame \
        --precision $PRECISION \


    python3 models/cox3d/main.py \
        --id "${MODEL}_${SIZE}_64_profile_${DATASET}_${DEVICE}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \
        --co3d_forward_mode frame \
        --temporal_window_size 64 \
        --precision $PRECISION \

done

SIZE=l
BATCH_SIZE=16


python3 models/cox3d/main.py \
    --id "${MODEL}_${SIZE}_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --precision $PRECISION \


python3 models/cox3d/main.py \
    --id "${MODEL}_${SIZE}_64_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --co3d_forward_mode frame \
    --temporal_window_size 64 \
    --precision $PRECISION \




MODEL="i3d"
BATCH_SIZE=16

python3 models/i3d/main.py \
    --id "${MODEL}_profile_${DATASET}" \
    --dataset $DATASET \
    --gpus 1 \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/i3d/hparams/i3d.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --precision $PRECISION \




MODEL="r2plus1d"
BATCH_SIZE=16

python3 models/r2plus1d/main.py \
    --id "${MODEL}_8_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --profile_model \
    --gpus $GPUS \
    --temporal_window_size 8 \
    --logging_backend $LOGGING_BACKEND \
    --precision $PRECISION \

python3 models/r2plus1d/main.py \
    --id "${MODEL}_16_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --profile_model \
    --gpus $GPUS \
    --temporal_window_size 16 \
    --logging_backend $LOGGING_BACKEND \
    --precision $PRECISION \




MODEL="slow"
BATCH_SIZE=8

python3 models/slow/main.py \
    --id "${MODEL}_profile_${DATASET}_${DEVICE}" \
    --from_hparams_file models/slow/hparams/slow_8x8.yaml \
    --dataset $DATASET \
    --seed 123 \
    --profile_model \
    --gpus $GPUS \
    --batch_size $BATCH_SIZE \
    --logging_backend $LOGGING_BACKEND \
    --precision $PRECISION \




MODEL="slowfast"
BATCH_SIZE=32

for SIZE in 4x16_R50 8x8_R50
do
    python3 models/slowfast/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}_${DEVICE}" \
        --dataset $DATASET \
        --seed 123 \
        --gpus $GPUS \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/slowfast/hparams/$SIZE.yaml \
        --logging_backend $LOGGING_BACKEND \
        --profile_model \
        --image_size 256 \
        --precision $PRECISION \

done




MODEL="x3d"
BATCH_SIZE=64

for SIZE in xs s m
do
    python3 models/x3d/main.py \
        --id "${MODEL}_${SIZE}_profile_${DATASET}_${DEVICE}" \
        --dataset $DATASET \
        --gpus $GPUS \
        --seed 123 \
        --batch_size $BATCH_SIZE \
        --from_hparams_file models/x3d/hparams/$SIZE.yaml \
        --profile_model \
        --logging_backend $LOGGING_BACKEND \
        --precision $PRECISION \


done

SIZE=l
BATCH_SIZE=32

python3 models/x3d/main.py \
    --id "${MODEL}_${SIZE}_profile_${DATASET}_${DEVICE}" \
    --dataset $DATASET \
    --gpus $GPUS \
    --seed 123 \
    --batch_size $BATCH_SIZE \
    --from_hparams_file models/x3d/hparams/$SIZE.yaml \
    --profile_model \
    --logging_backend $LOGGING_BACKEND \
    --precision $PRECISION \
