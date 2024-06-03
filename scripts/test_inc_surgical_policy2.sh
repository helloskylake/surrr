#conda activate surrol_aloha
cd ..
# dual arm short time and chunk size
for model in Diffusion
do
  for task_name in NeedleRegrasp-v0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-inc/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --state_dim 14 --action_dim 10 --num_steps 50000 --lr 15e-6 --seed 0 --is_surgical  --eval_every 1250 --save_every 2500 --temporal_agg
done
done

# single arm long time
for model in Diffusion
do
  for task_name in  PegTransfer-v0 NeedlePick-v0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-inc/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 50000 --lr 15e-6 --seed 0 --is_surgical  --eval_every 1250 --save_every 2500 --temporal_agg
done
done

# single arm short time
for model in Diffusion
do
  for task_name in  NeedleReach-v0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-inc/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 5000 --lr 15e-6 --seed 0 --is_surgical  --eval_every 1250 --save_every 2500 --temporal_agg
done
done

# dual arm long time and chunk_size
for model in Diffusion
do
  for task_name in  BiPegTransfer-v0
do
  for chunk_size in  40
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-inc/ --policy_class $model --kl_weight 200 --chunk_size $chunk_size --hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --state_dim 14 --action_dim 10 --num_steps 50000 --lr 15e-6 --seed 0 --is_surgical  --eval_every 1250 --save_every 2500 --temporal_agg
done
done
done


