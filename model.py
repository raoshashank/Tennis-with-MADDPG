import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim) 

class Actor(nn.Module):
    def __init__(self,state_size,action_size,seed):
        # defines the deterministic policy : S-> A
        super(Actor,self).__init__()
        
        #hidden unit layer as given in paper
        fc_units = [256,128]
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size,fc_units[0])
        self.fc2 = nn.Linear(fc_units[0],fc_units[1])
        self.fc3 = nn.Linear(fc_units[1],action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        #This comes from Experimental Details section of paper.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #Last layer weights were initialized between -3e-3 and 3e3
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        #self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        
        
    def forward(self,state):
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        #return F.tanh(self.fc3(x))
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
    
    
class Critic(nn.Module):
    def __init__(self,state_size,action_size,seed):
        # defines the deterministic policy : S-> A
        super(Critic,self).__init__()
        
        #hidden unit layer as given in paper
        fc_units = [256,128]
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        state_size =48
        action_size= 4
        #self.fc1 = nn.Linear(state_size,fc_units[0])
        #self.fc2 = nn.Linear(fc_units[0]+action_size,fc_units[1])
        self.fc1 = nn.Linear(state_size+action_size,fc_units[0])
        self.fc2 = nn.Linear(fc_units[0],fc_units[1])
        self.fc3 = nn.Linear(fc_units[1],1)
        self.reset_parameters()
    
    def reset_parameters(self):
        #This comes from Experimental Details section of paper.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #Last layer weights were initialized between -3e-3 and 3e3
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        #self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        
        
    def forward(self,state,actions):
        #print(np.shape(state))
        #print(np.shape(actions))
        #print(actions)
        #a = input()
        x = torch.cat((state,actions),dim=1)
        x = F.relu(self.fc1(x))
        #x = torch.cat((x,actions),dim=1)
        x=F.relu(self.fc2(x))
        return self.fc3(x)
        

    
    