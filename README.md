# Imitation Learning algorithms for surgical task automation


#### 

This repo contains the implementation of ACMT, ACT, Diffusion Policy and CNNMLP, it's designed to train surgical robots with imitation learning algorithm.


### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate policy
- ``record_surgical_episodes.py`` collect expert demonstration with delta control
- ``record_surgical_episodes_joint.py`` collect expert demonstration with absolute joint control
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions



### Installation

    conda create -n sur python=3.8.10
    conda activate sur
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`(clone this into the main directory of project)
- also need to install https://github.com/med-air/SurRoL/tree/SurRoL-v2 for pybullet surgical environment by `pip install -e .` (clone this outside the main directory and just install in conda env)
- also need to install https://github.com/state-spaces/mamba/tree/main
- edit the gym in conda env by adding `import surrol.gym` 
### Example Usages

To set up a new terminal, run:

    conda activate sur
    cd <path to sur repo>

### Simulated experiments (LEGACY table-top ALOHA environments)
To collect data
    
    # Collect expert demonstration
    python3 --task_name NeedlePick-v0 --dataset_dir /dir --num_episodes 100 --onscreen_render
To train ACMT follow the scripts in folder and modify the mamba enable flag in `transformer.py`:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name NeedlePick-v0 --ckpt_dir <ckpt dir> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --num_epochs 100000  --lr 1e-5 --seed 0 --is_surgical --is_joint



