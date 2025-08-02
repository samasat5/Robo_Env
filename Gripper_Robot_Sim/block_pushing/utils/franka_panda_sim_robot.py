# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XArm Robot Kinematics."""
from block_pushing.utils import utils_pybullet
from block_pushing.utils.pose3d_gripper import Pose3d_gripper
import numpy as np
from scipy.spatial import transform
import pybullet
import pdb
import time

XARM_URDF_PATH = (
    "third_party/bullet/examples/pybullet/gym/pybullet_data/" "xarm/xarm6_robot.urdf"
)
SUCTION_URDF_PATH = "third_party/py/envs/assets/suction/" "suction-head-long.urdf"
CYLINDER_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder.urdf"
CYLINDER_REAL_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder_real.urdf"
HOME_JOINT_POSITIONS = np.deg2rad([0, -30, 0, -60, 0, 90, 0, 0.04, 0.04]) # set only for the 9 joints (revolu + per)
PANDA_URDF_PATH = ("third_party/bullet/examples/pybullet/gym/pybullet_data/" "franka_panda/panda.urdf")    # Khodam
physics_client = p.connect(p.GUI)  # Use GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1. / 240.)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-40,
    cameraTargetPosition=[0, 0, 0.1],
)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Create robot
INITIAL_JOINT_POSITIONS = np.array(
    [
        0.0, 
        -0.5235987755982988, 
        0.0, -1.0471975511965976, 
        0.0, 1.5707963267948966, 
        0.0, 
        0.0, 
        0.0])
BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"
workspace_uid = utils_pybullet.load_urdf(
    p,
    WORKSPACE_URDF_PATH,
    basePosition=[0.35, 0, 0.0],)

robot = GripperArmSimRobot(p,INITIAL_JOINT_POSITIONS)
target_id = utils_pybullet.load_urdf(p,ZONE_URDF_PATH,
                                      [0.4999, -0.36, 0.1], 
                                      useFixedBase=True)
block_id = utils_pybullet.load_urdf(p, BLOCK_URDF_PATH, 
                                     [0.2, 0.47, 0.01],
                                     useFixedBase=False)
        
        
        
    def set_target_pick_n_place_the_block (self, place_position, block_position):
        self.set_target_pick_the_block(block_position)
        # Move above the placement target
        hover_position = place_position +np.array([0.35, 0, -place_position[2]+0.15])
        self.move_gripper_to_target(hover_position)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
        # Move down to place
        place_position_z = place_position +np.array([0.35, 0, -place_position[2]-0.0001])
        self.move_gripper_to_target(place_position_z)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
        
    