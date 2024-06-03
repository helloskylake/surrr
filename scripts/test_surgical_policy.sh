#conda activate surrol_aloha
cd ..
#for model in ACT
#do
#  for task_name in  NeedleRegrasp-v0
#do
#  for chunk_size in  20
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name/ --policy_class $model --kl_weight 200 --chunk_size $chunk_size --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 14 --action_dim 14 --num_steps 100000 --lr 75e-7 --seed 0 --is_surgical --is_joint --eval_every 2500 --save_every 5000 --temporal_agg
#done
#done
#done

#for model in ACT
#do
#  for task_name in NeedleRegrasp-v0  BiPegTransfer-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 14 --action_dim 14 --num_steps 100000 --lr 75e-7 --seed 0 --is_surgical --is_joint --eval_every 2500 --save_every 5000 --temporal_agg
#done
#done

# single arm
for model in ACT
do
  for task_name in  NeedlePick-v0  # NeedleReach-v0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 100000 --lr 75e-7 --seed 0 --is_surgical --is_joint --eval_every 2500 --save_every 5000 --temporal_agg
done
done