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

gan_past_length=10
export gan_past_length
echo $gan_past_length


ulimit -n 4096

num_dataloader_workers_trainer=32
num_dataloader_workers_runner=16

# For debug.
# TODO: add argument handling

cache_dir=~/cache_mgh/img_cache
export cache_dir

python $sdtm_home/scripts/train_temporal_model.py  --data_dir ${data_dir} --annotation_filename ${annotation_dir} --log_dir $sdtm_home/logs_gan --learning_rate 4e-5 --mse_coeff 0.2 --training_ratio 1.0 --gan_past_length $gan_past_length --gan_future_length $gan_future_length --sampling_rate $sampling_rate --model_filename $sdtm_home/temporal_weights/${script_arguments}_statedict.pt --phase_order_filename $sdtm_home/mgh_phase_order.json --phase_length_filename $sdtm_home/LapChole_phase_length_mgh100.json --cache_dir $cache_dir --num_dataloader_workers ${num_dataloader_workers_trainer} --batch_size 8
