import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import gym
import imageio

from surrol.const import ROOT_DIR_PATH
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS, surgical_tasks_joint
from collections import OrderedDict
from utils import parse_ts
import IPython
e = IPython.embed



def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    render_cam_name = 'ecm'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = surgical_tasks_joint[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    is_bimanul = surgical_tasks_joint[task_name]['is_bimanual']


    success = []
    episode_idx = 0
    while episode_idx < num_episodes:
        onscreen_render = 1
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = gym.make(task_name)
        seeds = np.random.randint(0, 2**32-1)
        
        env.seed(seeds)
        np.random.seed(seeds)
        ts = env.reset()  
        obs = ts
        ts = parse_ts(ts, env, is_joint=True, is_bi=is_bimanul)
        episode = []  
        # setup plotting
        view = 'ecm'  # ecm top front
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            # #record image from different view
            # image_ecm = ts.observation['images']['ecm']
            # imageio.imwrite('ecm.jpg', image_ecm)
            # image_front = ts.observation['images']['front']
            # imageio.imwrite('front.jpg', image_front)
            # image_top = ts.observation['images']['top']
            # imageio.imwrite('top.jpg', image_top)
            plt.title("scripted")
            plt.ion()
        for step in range(episode_len):
            action = env.get_oracle_action(obs)
            # print(action[4]) #qpos_psm1(0, 0, 0.1, 0, 0, 0)
            # psm1.get_current_joint_position [0.185499248417839, -0.007158239633729872, 0.14099653468572, -0.34351145262809457, -0.0563855950020197, -0.1769601211409867]
            # pose psm1 ((0.05, 0.24, 0.8524), (0, 0, -1.9198621771937625))
            # a = env.QPOS_PSM1
            # b = env.psm1.get_current_joint_position()
            # c = env.psm1.get_current_position()
            # d = [a[i]-b[i] for i in range(len(a))]
            # print(d)

            ts = env.step(action)  # psm env._set_action
            obs = ts[0]
            ts = parse_ts(ts, env, action, True, is_bi=is_bimanul)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode])  
        episode_max_reward = np.max([ts.reward for ts in episode])  
        
        if episode_max_reward == surgical_tasks_joint[task_name]['max_reward']:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]




        # clear unused variables
        del episode
        del env
        

        onscreen_render = 0
        # setup the environment
        print('Replaying joint commands')
        
        env = gym.make(task_name)
        env.seed(seeds)
        np.random.seed(seeds)
        ts = env.reset()
        ts = parse_ts(ts, env, is_bi=is_bimanul)
        episode_replay = [ts]  
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            plt.title("EE")
            plt.ion()
        # ts = env.step(action)
        for t in range(len(joint_traj)):
            action = joint_traj[t]
            # print(action)
            #
            ts = env.step_ee(action)
            ts = parse_ts(ts, env, action, is_joint=True, is_bi=is_bimanul)
            # print((np.array(ts.observation['qpos'])-np.array(action))[4:])
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.02)
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        print('episode reward/ env reward: ', episode_max_reward, surgical_tasks_joint[task_name]['max_reward'])
        if episode_max_reward == surgical_tasks_joint[task_name]['max_reward']:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")
            env.close()
            continue

        plt.close()

        #




        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        # joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)  
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])  
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        if task_name in surgical_tasks_joint.keys():
            robo_state_dim = surgical_tasks_joint[task_name]['state_dim']
            dim = surgical_tasks_joint[task_name]['action_dim']
        else:
            print('cant recognize task name')

        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')  
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )  
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, robo_state_dim))  
            action = root.create_dataset('action', (max_timesteps, dim))
            
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        env.close()
        episode_idx += 1
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

