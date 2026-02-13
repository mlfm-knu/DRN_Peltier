# PPO-based temperature control in thermoelectric heat exchange system
**This repository contains code for training a PPO (Proximal Policy Optimization) agent to perform real-time temperature control of a thermoelectric (Peltier) heat-exchange system.**

---

## Files

The repository contains the following files:

- `Peltier_env1.py`  
  This file defines the **reinforcement learning environment** designed for training a temperature control model.  
  It **loads pre-trained ANN-based weights (`500_timestep.h5`)** to approximate the system dynamics, and the agent learns the **PWM control action** through reinforcement learning.

- `Peltier_run1.py`  
  This is the **execution (run) script** for training the reinforcement learning–based temperature control model.  
  When executed, the script automatically creates the following directories:
  - `model/` : Stores the model weights saved during training  
  - `logs_Peltier1/` : Stores training results and log data (e.g., rewards, performance metrics)

- `500_timestep.h5`  
  Pre-trained ANN-based weights used to construct the reinforcement learning environment and initialize the system behavior model.

## Usage

### - Training: python Peltier_run1.py


## Output Directories

When training is executed, the following directories are automatically created:

- `model/`  
  Stores intermediate and final **PPO model weights** saved during training.

- `logs_Peltier1/`  
  Stores **training logs**, including rewards, temperature tracking performance, and other learning metrics.


## Dependencies

The following libraries are used in this project:

- numpy  
- pandas  
- torch  
- gymnasium  
- stable-baselines3  
- os  
- time  
- math 

### Version Info

- torch: 2.0.1+cu117  
- gymnasium: 0.29.1  
- stable-baselines3: 2.1.0  

## License
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

### Additional Information
- This program is registered software (Copyright Registration No. C-2026-008133, Korea Copyright Commission).
- Certain parts of the algorithm are protected under Patent No. 10-XXXX (KIPO).
- Copyright (c) 2024 MLFM Lab, Kyungpook National University.
- Some codes in this repository are modified from Dual deep mesh prior.

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and limitations under the License.

## Citation & Acknowledgment
If you use this code for your research, please cite the following paper:

**Deep reinforcement learning-based real-time temperature control in thermoelectric heat exchange system**  
Seokyong Lee, Ngan-Khanh Chau, Sanghun Choi  
DOI: https://doi.org/10.1016/j.csite.2025.107633
