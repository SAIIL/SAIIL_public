sdtm_home=~/SAIIL_public/src
export sdtm_home

# data_dir=~/SAIIL_public/data/videos/deidentified/laparoscopic_cholecystectomy/
data_dir=~/SAIIL_public/data/cholec80/videos_processed
# annotation_dir=~/SAIIL_public/data/annotations/deidentified/mgh_protobuf/protobufs
annotation_dir=~/SAIIL_public/data/annotations/deidentified//cholec80_protobuf

script_arguments=gan_past10_future10_sr0d2
export script_arguments

sampling_rate=0.2
export script_arguments
echo $script_arguments

ulimit -n 4096

num_dataloader_workers_trainer=32
num_dataloader_workers_runner=16

# For debug.
# TODO: add argument handling

cache_dir=~/cache_mgh/img_cache
export cache_dir

python $sdtm_home/scripts/train_temporal_model.py --track_name phase --data_dir ${data_dir} --annotation_filename ${annotation_dir}/train/ --log_dir $sdtm_home/logs_gan_fix_names --learning_rate 4e-5 --training_ratio 1.0 --temporal_length $gan_past_length --sampling_rate $sampling_rate --model_filename $sdtm_home/temporal_weights/${script_arguments}_statedict.pt --phase_order_filename $sdtm_home/mgh_phase_order.json --cache_dir $cache_dir --num_dataloader_workers ${num_dataloader_workers_trainer} --batch_size 8
