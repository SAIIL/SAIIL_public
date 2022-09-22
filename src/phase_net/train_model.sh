data_dir=your_video_path
anno_dir=your_annotation_path
sampling_rate=1.0
exp_name=cholec80

python ./train_baseline.py --track_name phase --train --inference --num_epoch 20 --gpu 0 --data_dir $data_dir  --annotation_filename $anno_dir --temporal_length 8 --sampling_rate $sampling_rate --cache_dir ./cache_mgh --log_dir ./lightning_logs/ --exp_name $exp_name --num_dataloader_workers 8 --batch_size 12