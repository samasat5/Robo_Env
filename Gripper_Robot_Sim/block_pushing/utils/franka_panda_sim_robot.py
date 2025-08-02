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

class GripperArmSimRobot:
    """A simulated PyBullet XArm robot, mostly for forward/inverse kinematics."""

    def __init__(
        self,
        pybullet_client,
        initial_joint_positions=HOME_JOINT_POSITIONS,
        end_effector="none",
        color="default",
    ):
        self._pybullet_client = pybullet_client
        self.initial_joint_positions = initial_joint_positions

        if color == "default":
            self.gripperarm = utils_pybullet.load_urdf(          # Khodam
                pybullet_client, PANDA_URDF_PATH, [0, 0, 0], useFixedBase = True
            )
        else:
            raise ValueError("Unrecognized xarm color %s" % color)

        # Get revolute joints of robot (skip fixed joints).
        joints = []
        joints_info_RevOrFixed = []
        joint_indices_revolutePrismatic = []
        
        for i in range(self._pybullet_client.getNumJoints(self.gripperarm)):        # Khodam
            joint_info = self._pybullet_client.getJointInfo(self.gripperarm, i)
            joints_info_RevOrFixed.append(joint_info[2])
            if joint_info[2] == pybullet.JOINT_REVOLUTE or joint_info[2] == pybullet.JOINT_PRISMATIC:
                joints.append(joint_info[0])
                joint_indices_revolutePrismatic.append(i)
                # Note examples in pybullet do this, but it is not clear what the
                # benefits are.
                self._pybullet_client.changeDynamics(
                    self.gripperarm, i, linearDamping=0, angularDamping=0
                )

        print("number of joints:",pybullet.getNumJoints(self.gripperarm))
        print("joint_indices:",joint_indices_revolutePrismatic)
        print("joints:",joints)
        print("joints_info_RevOrFixed:",joints_info_RevOrFixed)

        self._n_joints = len(joints)
        self._joints = tuple(joints)
        self._joint_indices = tuple(joint_indices_revolutePrismatic)

        # Move robot to home joint configuration
        self.reset_joints(self.initial_joint_positions)
        self.effector_link = 6
        self.gripper_target = 11  # Khodam
        self.right_finger = 9   # Khodam
        self.left_finger = 10   # Khodam
        self.grabbing_size_for_block = 0.0001
        self.block_size = 0.04
        self.closing_width = -0.005
        self.opening_width = self.block_size + self.grabbing_size_for_block
        self.offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
        
    def get_joint_positions(self):
        joint_states = self._pybullet_client.getJointStates(
            self.gripperarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        return joint_positions

    def _get_current_gripper_translation(self):
        state_right_finger = self._pybullet_client.getLinkState(self.gripperarm, self.right_finger)
        state_left_finger = self._pybullet_client.getLinkState(self.gripperarm, self.left_finger)
        translation_left = state_left_finger[0] 
        translation_right = state_right_finger[0]
        return [translation_left,
                translation_right]
    def _get_current_gripper_orientation(self):
        state_right_finger = self._pybullet_client.getLinkState(self.gripperarm, self.right_finger)
        state_left_finger = self._pybullet_client.getLinkState(self.gripperarm, self.left_finger)
        orientation_left = state_left_finger[1]
        orientation_right = state_right_finger[1]
        return [orientation_left,
                orientation_right]
        

    def reset_joints(self, joint_values):
        """Sets the position of the Robot's joints.

        *Note*: This should only be used at the start while not running the
                simulation resetJointState overrides all physics simulation.

        Args:
          joint_values: Iterable with desired joint positions.
        """

        for i in range(self._n_joints):  
            self._pybullet_client.resetJointState(
                self.gripperarm, self._joints[i], joint_values[i]
            )

    def get_joints_measured(self):
        joint_states = self._pybullet_client.getJointStates(   #Khodam
            self.gripperarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_torques = np.array([state[3] for state in joint_states])
        return joint_positions, joint_velocities, joint_torques


    def forward_kinematics(self):
        """Forward kinematics."""
        # effector_state = self._pybullet_client.getLinkState(
        #     self.xarm, self.effector_link
        # )
        right_finger_state = self._pybullet_client.getLinkState(    # khodam
            self.gripperarm, self.right_finger
        )
        left_finger_state = self._pybullet_client.getLinkState(    # khodam
            self.gripperarm, self.left_finger
        )
        # return Pose3d(
        #     translation=np.array(effector_state[0]),
        #     rotation=transform.Rotation.from_quat(effector_state[1]),
        # )
        return Pose3d_gripper(            # Khodam
            translation_left=np.array(left_finger_state[0]),
            translation_right=np.array(right_finger_state[0]),
            orientation_left=transform.Rotation.from_quat(left_finger_state[1]),
            orientation_right=transform.Rotation.from_quat(right_finger_state[1]),
        )
    def get_center_translation(self,translation_left,translation_right):   # Khodam
        """Compute midpoint between left and right finger poses."""
        center_translation = (translation_left + translation_right) / 2
        return center_translation
    
    def get_center_rotation(self,translation_left,translation_right):     # Khodam
        """ building a "default" rotation for the center of the gripper """
        
        x_axis = (translation_right - translation_left) # there r two fingers=> the line between them tells us the direction of the gripper’s opening
        x_axis /= np.linalg.norm(x_axis)

        # You need a grasping direction — either from a previous reference or default.
        z_axis = np.array([0, 0, -1])  # Let's say it's "downward" in world coordinates:

        # Recompute z to be orthogonal to x
        z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
        z_axis /= np.linalg.norm(z_axis)

        # Then y is x × z
        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to Rotation object
        center_rotation = transform.Rotation.from_matrix(rot_matrix)
        return center_rotation


    def inverse_kinematics(
        self, new_pose, max_iterations=100, residual_threshold=1e-10
    ):
        """Inverse kinematics.

        Args:
          world_effector_pose: Target Pose3d for the robot's end effector.
          max_iterations: Refine the IK solution until the distance between target
            and actual end effector position is below this threshold, or the
            maxNumIterations is reached. Default is 20 iterations.
          residual_threshold: Refine the IK solution until the distance between
            target and actual end effector position is below this threshold, or the
            maxNumIterations is reached.

        Returns:
          Numpy array with required joint angles to reach the requested pose.
        """
        translation_left = new_pose.translation_left
        translation_right = new_pose.translation_right
        center_translation = self.get_center_translation(translation_left,translation_right)
        center_rotation = self.get_center_rotation(translation_left,translation_right)
        return np.array( # 
            self._pybullet_client.calculateInverseKinematics(
            bodyUniqueId=self.gripperarm,
            endEffectorLinkIndex=self.gripper_target, 
            targetPosition=center_translation,
            targetOrientation=center_rotation.as_quat(),  # as_quat returns xyzw.
                lowerLimits=[-2.9] * 9, # : use 9 instead of 6
                upperLimits=[2.9] * 9,
                jointRanges=[5.8] * 9,
                restPoses=[0, 0] + self.get_joint_positions()[2:].tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
            )
        )

    def set_target_effector_pose(self, new_pose,force):  
        target_joint_positions = self.inverse_kinematics(new_pose)  
        print("target_joint_positions:", target_joint_positions)
        self.set_target_joint_positions(target_joint_positions,force)

        
    def set_target_joint_positions(self, target_joint_positions,force):
        # print("Moving to the new pose...")
        self._pybullet_client.setJointMotorControlArray(
            self.gripperarm, 
            self._joint_indices,
            pybullet.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[force * 240.0] * len(self._joint_indices),
        )
    def set_the_fingers_open_close(self,opening_or_closing_width,force):
        print("Opening fingers...and wait")
        half_opening = opening_or_closing_width / 2.0
        
        self._pybullet_client.setJointMotorControlArray(      # Using POSITION_CONTROL
            bodyUniqueId=self.gripperarm,
            jointIndices=[self.left_finger, self.right_finger],
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=[half_opening, half_opening],
            forces=[force*100, force*100],            # Apply up to 100N of torque
            positionGains=[0.4, 0.4],     # Optional: reduce control stiffness
            velocityGains=[1.0, 1.0],    
        )



    def set_target_joint_velocities(self, target_joint_velocities):
        self._pybullet_client.setJointMotorControlArray(
            self.gripperarm,          #Khodam
            self._joint_indices,
            pybullet.VELOCITY_CONTROL,
            targetVelocities=target_joint_velocities,
            forces=[5 * 240.0] * len(self._joint_indices),
        )
    def set_alpha_transparency(self, alpha):
        visual_shape_data = self._pybullet_client.getVisualShapeData(self.gripperarm)

        for i in range(self._pybullet_client.getNumJoints(self.gripperarm)):
            object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
            assert object_id == self.gripperarm, "xarm id mismatch."
            assert link_index == i, "Link visual data was returned out of order."
            rgba_color = list(rgba_color[0:3]) + [alpha]
            self._pybullet_client.changeVisualShape(
                self.gripperarm, linkIndex=i, rgbaColor=rgba_color
            )
            



    def move_gripper_to_target (self,target_center,force):
        pose = self.forward_kinematics()
        
        
        new_translation_left = target_center - self.offset
        new_translation_right = target_center + self.offset
        center_rotation = self.get_center_rotation(new_translation_left, new_translation_right)
        new_pose = Pose3d_gripper(translation_left=new_translation_left,
                                translation_right=new_translation_right,
                                orientation_left=center_rotation, 
                                orientation_right=center_rotation) 
        print("_____________Force is 1")
        
        self.set_target_effector_pose(new_pose,force)


    
    def set_target_pick_the_block(self, block_position):
        force = 2
        opening_width = self.opening_width
        self.set_the_fingers_open_close(opening_width,force)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
            
        force = 7
        self.move_gripper_to_target (block_position, force)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)

        force = 1
        closing_width = self.closing_width
        self.set_the_fingers_open_close(closing_width,force)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
        
        
        
    def set_target_pick_n_place_the_block (self, place_position, block_position):
        self.set_target_pick_the_block(block_position)
        # Move above the placement target
        force = 0.4
        feasible_place_position = place_position + np.array([0.0, 0, 0.15])
        self.move_gripper_to_target(place_position,force)
        for _ in range(1000):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
        # Move down to place
        place_position_z = place_position +np.array([0.35, 0, -0.01])
        force = 0.2
        self.move_gripper_to_target(place_position_z,force)
        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(1 / 240.0)
        
    