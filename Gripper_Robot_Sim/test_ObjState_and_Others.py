import pybullet as p
import pybullet_data
import time
from typing import Tuple, Any

from block_pushing.utils.utils_pybullet import ObjState,XarmState
from block_pushing.utils.pose3d_gripper import Pose3d_gripper
import numpy as np

# === First, Setup PyBullet ===
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf",[0, 0, -0.001])

p.resetDebugVisualizerCamera(
    cameraDistance=1,
    cameraYaw=100,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.1],
)

obj_id = p.loadURDF("franka_panda/panda.urdf",[0.2, 0.5, 0.01]  , useFixedBase = False ) # size="0.04 0.04 0.04



# === Step a few frames ===
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240)

# === 1. Save state ===
state = ObjState.get_bullet_state(p, obj_id)
inf = ObjState._get_joint_info(p, obj_id, 9)
print("info:", inf)
print("[Saved State]")
# ObjState(
#     obj_id: int,
#     base_pose: Tuple[Vec3, Vec4],    # Position (xyz) and orientation (quaternion) of the base
#     base_vel: Tuple[Vec3, Vec3],     # Linear and angular velocity of the base
#     joint_info: Any,                 # Metadata (like names, limits, types) for each joint
#     joint_state: Any                 # Position, velocity, torque, etc. for each joint
# )
print("Base pose:", state.base_pose)
print("Joint positions:", [js[0] for js in state.joint_state])




target_center = np.array([0.2, 0.5, 0.01 + 0.015])
offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
new_translation_left = target_center - offset
new_translation_right = target_center + offset
 
new_pose = Pose3d_gripper(translation_left=new_translation_left,
                          translation_right=new_translation_right,
                          rotation_left=pose.rotation_left, 
                          rotation_right=pose.rotation_left)
state = XarmState.get_bullet_state(client, obj_id, target_pose, goal_translation)


# === 2. Apply random velocity (modify the object) ===
p.resetBaseVelocity(obj_id, linearVelocity=[1, 0, 0], angularVelocity=[0, 0, 0])
for _ in range(50):
    p.stepSimulation()
    time.sleep(1 / 240)

# === 3. Restore state ===
state.set_bullet_state(p, obj_id)

# === 4. Check restored state ===
restored_pose = p.getBasePositionAndOrientation(obj_id)
print("\n[Restored State]")
print("Restored pose:", restored_pose)
print("Should match saved pose:", state.base_pose)

# === 5. Test Serialization ===
serialized = state.serialize()
print("\nSerialized:", serialized.keys())

# === 6. Test Deserialization ===
deserialized = ObjState.deserialize(serialized)
assert deserialized.base_pose == state.base_pose

