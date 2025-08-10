# ################################
# Move a truck robot forword in PYBULLET
# ################################

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p 
import pybullet_data 
import time


p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=2.0)

robot_id = p.loadURDF("husky/husky.urdf",[0,0,0], [0,0,0,1], useFixedBase = False )

p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=50,
    cameraPitch=-50,
    cameraTargetPosition=[0.5, 0, 0.5],
)

time.sleep(3)

nb_joints = p.getNumJoints(robot_id)
print(nb_joints)


# Adding force to the four wheels: 
p.setJointMotorControlArray(       # Using VELOCITY_CONTROL
    bodyUniqueId=robot_id,
    jointIndices=[2, 3, 4, 5],
    controlMode=p.VELOCITY_CONTROL,
    targetVelocities=[3.0, 3.0, 3.0, 3.0],  # rad/sec
    forces=[20, 20, 20, 20]                
)
# Step the simulation
for _ in range(240):
    p.stepSimulation()
    time.sleep(1 / 240.0)


p.setJointMotorControlArray(      # Using POSITION_CONTROL
    bodyUniqueId=robot_id,
    jointIndices=[2,3,4,5],
    controlMode=p.POSITION_CONTROL,
    targetPositions=[10, 10,10,10],
    forces=[100, 100,100,100],            # Apply up to 100N of torque
    positionGains=[0.4, 0.4,0.4, 0.4],     # Optional: reduce control stiffness
    velocityGains=[1.0, 1.0,1.0, 1.0],    
)

for _ in range(240):
    p.stepSimulation()
    time.sleep(1 / 240.0)



p.setJointMotorControlArray(       
    bodyUniqueId=robot_id,
    jointIndices=[2, 3, 4, 5],
    controlMode=p.VELOCITY_CONTROL,
    targetVelocities=[-3.0, -3.0, -3.0, -3.0],  # rad/sec
    forces=[20, 20, 20, 20]                
)
# Step the simulation
for _ in range(240):
    p.stepSimulation()
    time.sleep(1 / 240.0)
    

p.setJointMotorControlArray(  # moving to the right
    bodyUniqueId=robot_id, 
    jointIndices=[2, 3, 4, 5],  
    controlMode=p.VELOCITY_CONTROL,
    targetVelocities=[-5.0, 5.0, -5.0, 5.0],  # Left wheels go backward, right forward
    forces=[40, 40, 40, 40]
)

# Step the simulation
for _ in range(240):
    p.stepSimulation()
    time.sleep(1 / 240.0)

print("done.")