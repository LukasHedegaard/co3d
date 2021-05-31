#!/bin/bash

PROJECT=models/x3d
DATASET=kinetics400

for MODEL in xs s m l
do

    horovodrun -np 4 python $PROJECT/main.py \
        --id "x3d_kinetics400_validate_center_sample_$MODEL" \
        --results_log_dir $PROJECT \
        --dataset $DATASET \
        --gpus 1 \
        --seed 42 \
        --batch_size 32 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
        --log_level DEBUG \
        --validate \
        --test_ensemble 0 \
        --distributed_backend horovod \
        --logging_backend wandb \
        --benchmark True \
        --forward_frame_delay 0 \

done