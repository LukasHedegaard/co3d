PROJECT=models/x3d
DATASET=kinetics600
GPUS="4"

python $PROJECT/main.py \
    --id x3d_xs \
    --results_log_dir $PROJECT \
    --dataset $DATASET \
    --gpus $GPUS \
    --max_epochs 100 \
    --seed 123 \
    --optimization_metric top1acc \
    --notify \
    --learning_rate 0.1 \
    --learning_rate_start_div_factor 100 \
    --batch_size 64 \
    --train \
    --test \
    --from_hparams_file $PROJECT/hparams/xs.yaml \

