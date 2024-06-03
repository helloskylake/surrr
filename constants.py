import pathlib
import os

surgical_tasks ={'NeedlePick-v0':{
        'action_dim': 5,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 50,  # for collect data
        'max_reward':2
    },
    'NeedleReach-v0':{
        'action_dim': 5,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 50,  # for collect data
        'max_reward':1
    },
    'NeedleRegrasp-v0':{
        'action_dim': 10,
        'state_dim': 14,
        'is_bimanual': True,
        'episode_len': 50,  # for collect data
        'max_reward':2
    },
    'PegTransfer-v0':{
        'action_dim': 5,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 100,  # for collect data
        'max_reward':3
    },
    'BiPegTransfer-v0':{
        'action_dim': 10,
        'state_dim': 14,
        'is_bimanual': True,
        'episode_len': 100,  # for collect data
        'max_reward':4
    }}
surgical_tasks_joint ={'NeedlePick-v0':{
        'action_dim': 7,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 50,  # for collect data
        'max_reward':2
    },
    'NeedleReach-v0':{
        'action_dim': 7,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 50,  # for collect data
        'max_reward':1
    },
    'NeedleRegrasp-v0':{
        'action_dim': 14,
        'state_dim': 14,
        'is_bimanual': True,
        'episode_len': 50,  # for collect data
        'max_reward':2
    },
    'PegTransfer-v0':{
        'action_dim': 7,
        'state_dim': 7,
        'is_bimanual': False,
        'episode_len': 100,  # for collect data
        'max_reward':3
    },
    'BiPegTransfer-v0':{
        'action_dim': 14,
        'state_dim': 14,
        'is_bimanual': True,
        'episode_len': 100,  # for collect data
        'max_reward':4
    }}
### Task parameters
DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else 'dataset_m/'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',   # '/sim_transfer_cube_scripted'
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']  
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_single_cube':{
        'dataset_dir': DATA_DIR + '/sim_single_cube',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'top']  
    },
    # #inc
    # 'NeedlePick-v0':{
    #     'dataset_dir': DATA_DIR + '/NeedlePick-v0-inc',
    #     'num_episodes': 100,
    #     'episode_len': 150,  # modify this
    
    # },
    # 'NeedleReach-v0':{
    #     'dataset_dir': DATA_DIR + '/NeedleReach-v0-inc',
    #     'num_episodes': 50,
    #     'episode_len': 50,  # modify this
    
    # },
    # 'NeedleRegrasp-v0':{
    #     'dataset_dir': DATA_DIR + '/NeedleRegrasp-v0-inc',
    #     'num_episodes': 100,
    #     'episode_len': 150,  # modify this
    
    # },
    # 'PegTransfer-v0':{
    #     'dataset_dir': DATA_DIR + '/PegTransfer-v0-inc',
    #     'num_episodes': 100,
    #     'episode_len': 150,  # modify this
    
    # },
    # 'BiPegTransfer-v0':{
    #     'dataset_dir': DATA_DIR + '/BiPegTransfer-v0-inc',
    #     'num_episodes': 100,
    #     'episode_len': 150,  # modify this
    
    # },
    # joint
    'NeedlePick-v0':{
        'dataset_dir': DATA_DIR + '/NeedlePick-v0-joint',
        'num_episodes': 100,
        'episode_len': 150,  # modify this
        'camera_names': ['ecm', 'top', 'front']  
    },
    'NeedleReach-v0':{
        'dataset_dir': DATA_DIR + '/NeedleReach-v0-joint',
        'num_episodes': 50,
        'episode_len': 50,  # modify this
        'camera_names': ['ecm', 'top', 'front']  
    },
    'NeedleRegrasp-v0':{
        'dataset_dir': DATA_DIR + '/NeedleRegrasp-v0-joint',
        'num_episodes': 100,
        'episode_len': 150,  # modify this
        'camera_names': ['ecm', 'top', 'front']  
    },
    'PegTransfer-v0':{
        'dataset_dir': DATA_DIR + '/PegTransfer-v0-joint',
        'num_episodes': 100,
        'episode_len': 150,  # modify this
        'camera_names': ['ecm', 'top', 'front']  
    },
    'BiPegTransfer-v0':{
        'dataset_dir': DATA_DIR + '/BiPegTransfer-v0-joint',
        'num_episodes': 100,
        'episode_len': 150,  # modify this
        'camera_names': ['ecm', 'top', 'front']  
    },

}

### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
START_ARM_POSE_S = [0.0869, 0.141, -0.0869, -1.72, 0.029, 2.49, 0.666, 0.04, 0.04]  

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800  
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# panda finger
PANDA_GRIPPER_POSITION_OPEN = 255  
PANDA_GRIPPER_POSITION_CLOSE = 0.0

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213  

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2


# for panda
PANDA_GRIPPER_POSITION_NORMALIZE_FN = lambda x: ((x - PANDA_GRIPPER_POSITION_CLOSE) / (PANDA_GRIPPER_POSITION_OPEN - PANDA_GRIPPER_POSITION_CLOSE))
PANDA_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: (x * (PANDA_GRIPPER_POSITION_OPEN - PANDA_GRIPPER_POSITION_CLOSE) + PANDA_GRIPPER_POSITION_CLOSE)