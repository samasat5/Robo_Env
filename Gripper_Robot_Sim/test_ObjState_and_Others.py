import pybullet as p
import pybullet_data
import time
from typing import Tuple, Any

from block_pushing.utils.utils_pybullet import ObjState

# === First, Setup PyBullet ===
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# === Load a sample object (KUKA robot or any URDF) ===
obj_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

# === Step a few frames ===
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240)

# === 1. Save state ===
state = ObjState.get_bullet_state(p, obj_id)
print("[Saved State]")
print("Base pose:", state.base_pose)
print("Joint positions:", [js[0] for js in state.joint_state])

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

print("\n[Deserialization works ✅]")

# Optional: Disconnect when done
# p.disconnect()
