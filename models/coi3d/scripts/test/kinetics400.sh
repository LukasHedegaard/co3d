#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coi3d
DATASET=kinetics400
PRECISION=32

# Run test sequence ######################


python $PROJECT/main.py \
    --id CoI3D_kinetics_frames_8 \
    --dataset $DATASET \
    --seed 42 \
    --gpus 1 \
    --from_hparams_file models/coi3d/hparams/i3d.yaml \
    --finetune_from_weights models/coi3d/weights/I3D_8x8_R50.pkl \
    --co3d_forward_mode init_frame \
    --batch_size 1 \
    --benchmark True \
    --num_workers 4 \
    --precision $PRECISION \
    --test \
    --limit_test_batches 2 \
    # --logging_backend wandb \


# python $PROJECT/main.py \
#     --id CoI3D_kinetics_frames_64 \
#     --dataset $DATASET \
#     --seed 42 \
#     --gpus 1 \
#     --from_hparams_file models/coi3d/hparams/i3d.yaml \
#     --finetune_from_weights models/coi3d/weights/I3D_8x8_R50.pkl \
#     --co3d_forward_mode init_frame \
#     --batch_size 1 \
#     --logging_backend wandb \
#     --num_workers 4 \
#     --precision $PRECISION \
#     --temporal_window_size 64 \
#     --test \
