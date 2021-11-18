PROJECT=models/x3d
DATASET=kinetics400

MODEL=s
for FRAME_DELAY in 0 8 16 24 32 40 47 #Max is 60-13=47
do

    horovodrun -np 4 python $PROJECT/main.py \
        --id CoX3D_transient_replicate_untrained_$MODEL \
        --dataset $DATASET \
        --seed 42 \
        --gpus 1 \
        --from_hparams_file models/x3d/hparams/$MODEL.yaml \
        --finetune_from_weights models/x3d/weights/x3d_$MODEL.pyth \
        --batch_size 32 \
        --forward_frame_delay $FRAME_DELAY \
        --benchmark True \
        --validate \
        --distributed_backend horovod \
        --logging_backend wandb \
        --num_workers 6 \
        --precision 16 \
        
done

MODEL=m
for FRAME_DELAY in 0 8 16 24 32 34 #Max is 50-16=34
do

    horovodrun -np 4 python $PROJECT/main.py \
        --id x3d_kinetics400_frame_delay_$MODEL \
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
        --forward_frame_delay $FRAME_DELAY \

        
done