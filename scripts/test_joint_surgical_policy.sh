#conda activate surrol_aloha
cd ..
#for model in ACT
#do
#  for task_name in NeedlePick-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-mamba/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 120000 --lr 75e-7 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done

# single arm
#for model in ACT
#do
#  for task_name in  PegTransfer-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 100000 --lr 75e-7 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done

#for model in ACT
#do
#  for task_name in  NeedleReach-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 20000 --lr 75e-7 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done


#for model in ACT
#do
#  for task_name in  BiPegTransfer-v0 # NeedleRegrasp-v0 #BiPegTransfer-v0
#do
#  for chunk_size in  20
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-mamba/ --policy_class $model --kl_weight 200 --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 14 --action_dim 14 --num_steps 150000 --lr 1e-5 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint #--resume_ckpt_path checkpoint_m/$model/$task_name-rope30/policy_last.ckpt
#done
#done
#done

for model in ACT
do
  for task_name in  PegTransfer-v0
do
  for kl_weight in  200
do
  for chunk_size in 30
do
  for seed in 42 66
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-kl$kl_weight/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 125000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
done
done
done
done
done

#for model in ACT
#do
#  for task_name in  NeedlePick-v0
#do
#  for kl_weight in  200
#do
#  for chunk_size in 5 10 30
#do
#  for seed in 0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-ck$chunk_size/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 120000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done
#done
#done
#done
