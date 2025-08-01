
import pybullet as p
import pybullet_data
import time
import numpy as np
import pdb
import math
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id  = p.loadURDF("franka_panda/panda.urdf",[0,0,0], [0,0,0,1], useFixedBase = True )
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-40,
    cameraTargetPosition=[0, 0, 0.1],
)
# information on the joints
movable_joints = []
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    joint_type = joint_info[2]
    joint_name = joint_info[1].decode("utf-8")
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]: # only the joints that move 
        movable_joints.append(i)
        p.addUserDebugText(
            f"{i}: {joint_name}",
            [0, 0, 0.1],
            parentObjectUniqueId=robot_id,
            parentLinkIndex=i,
            textColorRGB=[1, 0, 0],
            textSize=1.0,)

# move the joints one by one
for joint_index in movable_joints:
    print(f"Moving joint {joint_index}")
    for t in range(240):  
        angle = 0.5 * math.sin(t*0.05) # using sin to have a oscillating movement 
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            force=400,)
        p.stepSimulation()
        time.sleep(1 / 240)


while True:
    p.stepSimulation()
    time.sleep(1)