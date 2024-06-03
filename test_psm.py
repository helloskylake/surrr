import os
import time
import numpy as np
from tqdm import tqdm

import pybullet as p
import pybullet_data
from surrol.utils.pybullet_utils import (
    step,
    get_joints,
    get_link_name,
    reset_camera,
)
from surrol.robots.psm import Psm

scaling = 1.

p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
reset_camera(yaw=90, pitch=-15, dist=0.9*scaling)

p.loadURDF("plane.urdf", [0, 0, -0.001], globalScaling=1)

psm = Psm((0, 0, 0.1524),
          p.getQuaternionFromEuler((0, 0, -90/180*np.pi)),
          scaling=scaling)
psm.reset_joint((-0.2, 0, 0.20, 0, 0, 1))
print(psm.get_current_joint_position())
psm.move_joint([-0.52359879, 0., 0.12, 0., 0., 0.])
step(0.3)
print(psm.get_current_joint_position())