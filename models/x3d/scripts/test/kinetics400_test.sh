#!/bin/bash

PROJECT=models/x3d
DATASET=kinetics400

MODEL=xs

horovodrun -np 4 python $PROJECT/main.py \
    --id "x3d_kinetics400_test_center_sample_$MODEL" \
    --dataset $DATASET \
    --gpus 1 \
    --seed 42 \
    --batch_size 64 \
    --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    --test \
    --test_ensemble 0 \
    --distributed_backend horovod \
    --logging_backend wandb \
    --benchmark True \
    --forward_frame_delay 0 \
    --num_workers 6 \

MODEL=s

horovodrun -np 4 python $PROJECT/main.py \
    --id "x3d_kinetics400_test_center_sample_$MODEL" \
    --dataset $DATASET \
    --gpus 1 \
    --seed 42 \
    --batch_size 64 \
    --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    --test \
    --test_ensemble 0 \
    --distributed_backend horovod \
    --logging_backend wandb \
    --benchmark True \
    --forward_frame_delay 0 \
    --num_workers 6 \

MODEL=m

horovodrun -np 4 python $PROJECT/main.py \
    --id "x3d_kinetics400_test_center_sample_$MODEL" \
    --dataset $DATASET \
    --gpus 1 \
    --seed 42 \
    --batch_size 32 \
    --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    --test \
    --test_ensemble 0 \
    --distributed_backend horovod \
    --logging_backend wandb \
    --benchmark True \
    --forward_frame_delay 0 \
    --num_workers 6 \

MODEL=l

horovodrun -np 4 python $PROJECT/main.py \
    --id "x3d_kinetics400_test_center_sample_$MODEL" \
    --dataset $DATASET \
    --gpus 1 \
    --seed 42 \
    --batch_size 16 \
    --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
    --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
    --test \
    --test_ensemble 0 \
    --distributed_backend horovod \
    --logging_backend wandb \
    --benchmark True \
    --forward_frame_delay 0 \
    --num_workers 6 \