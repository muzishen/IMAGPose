 accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --use_deepspeed --num_processes 8 \
   --deepspeed_config_file zero_stage2_config.json \
   --deepspeed_multinode_launcher standard \
   train_stage2.py