#conda activate surrol_aloha
cd ..

for model in ACT
do
  for task_name in  BiPegTransfer-v0
do
  for chunk_size in 20
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope20/ --policy_class $model --kl_weight 200 --chunk_size $chunk_size --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --state_dim 14 --action_dim 14 --num_steps 100000 --lr 1e-5 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint --eval
done
done
done


