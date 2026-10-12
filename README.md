![Bottowork](bottowork.png)
# Real 2 Sim repo 

An overview of the repo. Collected episodes on real S0101 arm thorough teleoperation. 

## Key Features
- Real S0101 teleoperation data. The dataset lives on [hugging face](bottowork/so101_act_pickplace_v1)
- ACT + Recovery policy would be trained on Mujoco Sim
- Frozen Act weights and recovery MLP head would be used to train panda 7DOF arm
- Cross embodiment capabilities
- Policy would be uploaded to hugging face

