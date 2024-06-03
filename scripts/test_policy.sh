cd ..
for model in ACT Diffusion
do
  for task_name in sim_transfer_cube_scripted sim_insertion_human
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name/ --policy_class $model --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 20000 --lr 1e-5 --seed 0 --temporal_agg
done
done