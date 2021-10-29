#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/cox3d
DATASET=kinetics400

MODEL=s
FRAMES_PER_CLIP=13
FORWARD_FRAME_DELAY=56 # 84 # (56 + 64) * 0.7

python $PROJECT/main.py \
    --id CoX3D_s_kinetics400_ft \
    --dataset $DATASET \
    --seed 42 \
    --from_hparams_file models/x3d/hparams/$MODEL.yaml \
    --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
    --train \
    --num_sanity_val_steps 0 \
    --max_epochs 2 \
    --batch_size 3 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --gradient_clip_val 0.5 \
    --learning_rate 0.01 \
    --discriminative_lr_fraction 0.005 \
    --optimization_metric top1acc \
    --test \
    --logging_backend wandb \
    --num_workers 5 \
    --frames_per_clip $FRAMES_PER_CLIP \
    --co3d_num_forward_frames 8 \
    --co3d_forward_mode init_frame \
    --co3d_forward_frame_delay $FORWARD_FRAME_DELAY  \
    --gpus 1 \
    # --distributed_backend ddp \