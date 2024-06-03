from dm_control import suite
from dm_control import viewer
import numpy as np

# env = suite.load(domain_name="humanoid", task_name="stand")
# action_spec = env.action_spec()
#
# # Define a uniform random policy.
# def random_policy(time_step):   # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)
#
# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)


from dm_control import mujoco

from assets.dm_render import DMviewer


physics = mujoco.Physics.from_xml_path('assets/new_panda/single_cube.xml') 

viewer = DMviewer(physics)
while physics.time() < 1000:
  physics.step()
  viewer.render()