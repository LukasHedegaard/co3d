PROJECT=models/x3d
DATASET=kinetics400

MODEL=s
for TEMPORAL_DOWNSAMPLING in 1 2 3 4 5 6
do

    horovodrun -np 4 python $PROJECT/main.py \
        --id x3d_frame_rate_$MODEL \
        --dataset $DATASET \
        --seed 42 \
        --gpus 1 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
        --batch_size 32 \
        --temporal_downsampling $TEMPORAL_DOWNSAMPLING \
        --benchmark True \
        --validate \
        --distributed_backend horovod \
        --logging_backend wandb \
        --num_workers 6 \
        --precision 16 \
        
done

MODEL=m
for TEMPORAL_DOWNSAMPLING in 1 2 3 4 5 6
do

    horovodrun -np 4 python $PROJECT/main.py \
        --id x3d_frame_rate_$MODEL \
        --dataset $DATASET \
        --gpus 1 \
        --seed 42 \
        --batch_size 16 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
        --validate \
        --test_ensemble 0 \
        --distributed_backend horovod \
        --logging_backend wandb \
        --benchmark True \
        --num_workers 6 \
        --temporal_downsampling $TEMPORAL_DOWNSAMPLING \
        
done

MODEL=l
for TEMPORAL_DOWNSAMPLING in 2 #1 2 3 4 5 6
do

    CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python $PROJECT/main.py \
        --id x3d_frame_rate_$MODEL \
        --dataset $DATASET \
        --gpus 1 \
        --seed 42 \
        --batch_size 8 \
        --from_hparams_file $PROJECT/hparams/$MODEL.yaml \
        --finetune_from_weights $PROJECT/weights/x3d_$MODEL.pyth \
        --validate \
        --test_ensemble 0 \
        --distributed_backend horovod \
        --logging_backend wandb \
        --benchmark True \
        --num_workers 6 \
        --temporal_downsampling $TEMPORAL_DOWNSAMPLING \
        
done

