cd ..
for task_name in NeedleRegrasp-v0 PegTransfer-v0 BiPegTransfer-v0 NeedleReach-v0
do
python -u record_surgical_episodes_joint.py \
--task_name $task_name --dataset_dir dataset_m/$task_name-joint/ --num_episodes 100
done
