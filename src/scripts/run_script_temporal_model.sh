SAIIL_PUBLIC=~/SAIIL_public/src

data_dir=~/SAIIL_public/data/cholec80/videos_processed
annotation_dir=~/SAIIL_public/data/annotations/cholec80_protobuf
cache_dir=~/cache_mgh/img_cache

script_arguments=cnn_lstm
echo $script_arguments

sampling_rate=0.2
ulimit -n 4096
num_dataloader_workers_trainer=32

python ${SAIIL_PUBLIC}/scripts/train_temporal_model.py --track_name phase --data_dir ${data_dir} --annotation_filename ${annotation_dir}/ --log_dir ${SAIIL_PUBLIC}/logs --learning_rate 4e-5 --training_ratio 1.0 --temporal_length 8 --sampling_rate $sampling_rate --model_filename ${SAIIL_PUBLIC}/temporal_weights/${script_arguments}_statedict.pt --cache_dir $cache_dir --num_dataloader_workers ${num_dataloader_workers_trainer} --batch_size 8
