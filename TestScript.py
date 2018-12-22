from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle
from collections import deque
import pickle
import torch
import sys
from Agent_1 import Agent

env = UnityEnvironment(file_name="Tennis_Linux/Tennis")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain_name)
# In[4]:


env_info = env.reset(train_mode=False)[brain_name]
env.reset(train_mode=False)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
agents = []
[agents.append(Agent(state_size = state_size, action_size = action_size,seed = i)) for i in range(num_agents)]

RANDOM_EPISODES = 300
agents[0].actor_local.load_state_dict(torch.load('checkpoint_actor_1_last.pth',map_location = 'cpu'))
agents[0].critic_local.load_state_dict(torch.load('checkpoint_critic_1_last.pth',map_location = 'cpu'))
agents[1].actor_target.load_state_dict(torch.load('checkpoint_actor_2_last.pth',map_location = 'cpu'))
agents[1].critic_target.load_state_dict(torch.load('checkpoint_critic_2_last.pth',map_location = 'cpu'))

x = input()

for _ in range(1):
    env_info = env.reset(train_mode = False)[brain_name]
    states = env_info.vector_observations      
    [agents[i].reset() for i in range(num_agents)]

    for t in range(1000):
            actions=np.zeros([1,num_agents*action_size])
            actions = np.clip(np.random.random([1,4]),-1,1)
            env_info_ = env.step(actions)[brain_name]
            next_states = env_info_.vector_observations
            rewards = env_info_.rewards
            dones = env_info_.local_done 
            #add all the experience to common replay buffer (Attempt #1)
            #for i in range(0,num_agents):
            states=next_states
            if np.any(dones):
                break




for _ in range(RANDOM_EPISODES):
    env_info = env.reset(train_mode = False)[brain_name]
    states = env_info.vector_observations      
    [agents[i].reset() for i in range(num_agents)]

    for t in range(1000):
            actions=np.zeros([1,num_agents*action_size])
            actions =np.concatenate([agents[i].act(states[i],ADD_NOISE = False) for i in range(2)], axis = 0)
            env_info_ = env.step(actions)[brain_name]
            next_states = env_info_.vector_observations
            states=next_states
            rewards = env_info_.rewards
            dones = env_info_.local_done 
            if np.any(dones):
                break
                