#conda activate surrol_aloha
cd ..

#for model in ACT
#do
#  for task_name in  NeedleReach-v0
#do
#  for kl_weight in  200
#do
#  for chunk_size in 20
#do
#  for seed in 0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-minc/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 20000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg
#done
#done
#done
#done
#done
#
#for model in ACT
#do
#  for task_name in  NeedleRegrasp-v0
#do
#  for kl_weight in  200
#do
#  for chunk_size in 20
#do
#  for seed in 0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-minc/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 14 --action_dim 10 --num_steps 20000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg
#done
#done
#done
#done
#done

#for model in ACT
#do
#  for task_name in  PegTransfer-v0 # NeedlePick-v0
#do
#  for kl_weight in  200
#do
#  for chunk_size in 20
#do
#  for seed in 0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-minc/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 120000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg
#done
#done
#done
#done
#done

for model in ACT
do
  for task_name in  BiPegTransfer-v0
do
  for kl_weight in  200
do
  for chunk_size in 20
do
  for seed in 0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-minc/ --policy_class $model --kl_weight $kl_weight --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 14 --action_dim 10 --num_steps 150000 --lr 75e-7 --seed $seed --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg
done
done
done
done
done

