import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from gymnasium import spaces
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sequence_length = 500
path = '500_timestep.h5'



I = 1

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(sequence_length * 4, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def action_pwm(action):
    if action == 0:
        pwm = 100

    elif action ==1:
        pwm = -100

    elif action == 2:
        pwm = 30
    
    elif action == 3:
        pwm = 60
    
    elif action == 4:
        pwm = -60
    else:
        pwm = 60
    
    return pwm

def ANN_Model(path):
    model = ANN().to(device)
    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model



model = ANN_Model(path)


class PeltierEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        global I
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low = np.array([0,0]), high= np.array([100,100]),shape=(2,),dtype=np.float64)

        self.time_step = 0
        self.coolant = 17

        if I % 2 == 0:
            self.T_outlet = 30
            self.T_goal = 40
            init_seq = ([[self.time_step,self.T_outlet,0,self.coolant]])
            
            for i in range(self.time_step+1,self.time_step+500):
                seq = ([[i,self.T_outlet,0,self.coolant]])
                init_seq = np.concatenate((init_seq,seq),axis = 0)
            
            
            self.seq = init_seq


        else:
            self.T_outlet = 30
            self.T_goal = 40

            init_seq = ([[self.time_step,self.T_outlet,0,self.coolant]])
            
            for i in range(self.time_step+1,self.time_step+500):
                seq = ([[i,self.T_outlet,0,self.coolant]])
                init_seq = np.concatenate((init_seq,seq),axis = 0)

            
            self.seq = init_seq
        
        self.pwm = 0
        self.peltier_length = 2500
        self.reward = 0
        self.result = []
        self.i = 0
        self.u = 0
        

        
    
    def step(self,action):

        global I

        self.truncated = False
        self.pwm = action_pwm(action)
        x = []
        x.append(self.seq[self.i:self.i+501,:])
        x = np.array(x)

        #print(f"seq : {self.seq.shape}")
        for i in range (1,6):
            
            x = torch.tensor(x,dtype=torch.float32).to(device)
            #print(x.shape)
            
            with torch.no_grad():
                predictions = model(x).cpu().numpy()
                #print(x[0,:,0])
                #print(f"prediction : {predictions.shape}")
            
            self.time_step = self.time_step + 1
            new_time = 499 + self.time_step
            new_time = np.array(new_time)
            new_time = new_time.reshape(-1,1)
        
            new_pwm = self.pwm
            new_pwm = np.array(new_pwm)
            new_pwm = new_pwm.reshape(-1,1)

            new_coolant = self.coolant
            new_coolant = np.array(new_coolant)
            new_coolant = new_coolant.reshape(-1,1)

            new_data = np.concatenate((new_time,predictions,new_pwm,new_coolant),axis=1)
            result_tuple = (new_time,predictions,new_pwm,new_coolant)
            self.result.append(result_tuple)

            self.seq = np.concatenate((self.seq,new_data),axis=0)
            
            self.i += 1
            x =[]
            x.append(self.seq[self.i:self.i+501,:])
            x = np.array(x)
            
        
        self.T_outlet = predictions[0,0]
        #print(self.T_outlet.shape)
        loss = self.T_outlet-self.T_goal
        loss = pow(loss,2)
        landa = 0.8
        
        reward = math.exp(-landa*loss)

        self.reward = reward

        self.observation = [self.T_outlet,self.T_goal]
        self.observation = np.array(self.observation)

        if self.time_step > self.peltier_length:
            result = pd.DataFrame(self.result)
            model_dir = "result/heating"
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir) 
                
            result.to_csv(f'result/cooling_log{I}.csv',index=False)
            #global I
            I = I + 1
            print(self.u)
            self.done = True
        
        else:
            self.done = False
        self.u = self.u+1
        info = {}
        print(f"Reward: {self.reward}")
        print(f"time step : {self.time_step}")

        

        return self.observation, self.reward, self.done, self.truncated, info
    
    def reset(self,seed=None, options = None):
        
        self.time_step = 0
        self.coolant = 17
        
        if I % 2 == 0:
            self.T_outlet = 30
            self.T_goal = 40
            init_seq = ([[self.time_step,self.T_outlet,0,self.coolant]])
            
            for i in range(self.time_step+1,self.time_step+500):
                seq = ([[i,self.T_outlet,0,self.coolant]])
                init_seq = np.concatenate((init_seq,seq),axis = 0)
            
            
            self.seq = init_seq


        else:
            self.T_outlet = 30
            self.T_goal = 40

            init_seq = ([[self.time_step,self.T_outlet,0,self.coolant]])
            
            for i in range(self.time_step+1,self.time_step+500):
                seq = ([[i,self.T_outlet,0,self.coolant]])
                init_seq = np.concatenate((init_seq,seq),axis = 0)


            
            self.seq = init_seq
        
        
        self.pwm = 0
        self.peltier_length = 2500
        self.reward = 0
        self.result = []
        self.i = 0
        self.u = 0
        

        self.observation = [self.T_outlet,self.T_goal]
        self.observation = np.array(self.observation)
        info = {}

        return self.observation, info




    
