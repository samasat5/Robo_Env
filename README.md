
# A Pybullet Gym environment for a Pick-and-Place task with a Franka Panda robot

This environment introduces a new manipulation task and a new robot model compared to the original block_pushing environment from the Diffusion Policy paper. [https://github.com/real-stanford/diffusion_policy/tree/main]





**Run the Environmnt**
Run the block_picking3.py environment on sample actions:

Create the conda environment with the required dependencies 
```
conda env create -f requirements.yaml
conda activate <newenv_robot>
```
Run the sample env
```
(newenv_robot) [Gripper_Robot_Sim] python test_GripperArmSimRobot4.py
```


![Demo](./Gripper_Robot_Sim/media/demo.gif)