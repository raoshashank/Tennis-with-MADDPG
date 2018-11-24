from collections import namedtuple,deque
import torch
import numpy as np 
import random
import copy
from model import Actor,Critic
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###Hyper parameters as given in Paper####
GAMMA = 0.995
TAU   = 0.001
BATCH_SIZE= 512
BUFFER_SIZE= int(1e5)
weight_decay_Q = 1e-6
##########################################

class Agent():
    def __init__(self,state_size,action_size,seed,num_agents):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #actor = policy : S-> A
        self.actor_local = Actor(state_size,action_size,seed).to(device)
        #critic : Q-values : (S,A)->Q_value
        self.critic_local = Critic(state_size,action_size,seed).to(device)
        
        self.actor_target = Actor(state_size,action_size,seed).to(device)
        self.critic_target = Critic(state_size,action_size,seed).to(device)
        
        #Learning rates as given in the paper 
        self.actor_optimizer =torch.optim.Adam(self.actor_local.parameters(),lr=1e-4) 
        self.critic_optimizer=torch.optim.Adam(self.critic_local.parameters(),lr=1e-3,weight_decay = weight_decay_Q)
        
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed)
        
        self.noise = OUNoise(action_size,seed)
    
    def add_experience(self,states,actions,rewards,next_states,dones):
        #Add experience to Replay Buffer
         self.memory.add(states,actions,rewards,next_states,dones)
     
    def act(self,state,noise=0):
        #forward pass on actor network and get action
        state = torch.from_numpy(state).float().to(device)
        #for forward pass without training, set to eval mode
        self.actor_local.eval()
        
        with torch.no_grad():            
            #cuda tensors cannot be converted to numpy
            #for i in range(self.num_agents):
            action = self.actor_local(state).cpu().data.numpy()
        #set back to training mode
        self.actor_local.train()
        #add OUNoise to action to aid exploration
        for i in range(self.num_agents):
               action[i]+=self.noise.sample()
        return np.clip(action,-1,1)
   
       
    def step(self,t):
        #Add s,a,r,s tuple to memory 
        #if replay memory has batch_size experiences, then learn
        if len(self.memory)> BATCH_SIZE:
            if t%20 == 0:
                for _ in range(0,10):
                    self.learn(self.memory.sample())
            else:
                pass
           
                     
    def learn(self,experience):
        global GAMMA,TAU
        #Extract sars from experience
        state,action,reward,next_state,done = experience
        ####update critic first#####
        #get next action
        next_action_target = self.actor_target(next_state)
        #calculate actual Q value
        Q_actual = reward + (GAMMA*self.critic_target(next_state,next_action_target)*(1-done))
        
        #calculate expected Q value
        Q_expected = self.critic_local(state,action)
        #calculate loss and bpp
        critic_loss = F.mse_loss(Q_expected,Q_actual)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #Clip grad (Attempt #2)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()
        
        ###update actor next#####
        next_action_local = self.actor_local(state)
        actor_loss = -self.critic_local(state,next_action_local).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #Clip grad (Attempt #2)
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),1)
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target,TAU)
        self.soft_update(self.actor_local, self.actor_target,TAU)                     

        
         
    
    def reset(self):
        self.noise.reset()
    
    def soft_update(self, local_model, target_model,TAU=1.0):
        #global TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)        
        
class ReplayBuffer():
    def __init__(self,action_size,buffer_size,batch_size,seed):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience",field_names = ["state","action","reward","next_state","done"])
        self.buffer = deque(maxlen = buffer_size)
        
    def sample(self):
        exp = random.sample(self.buffer,k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in exp if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exp if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exp if e is not None])).float().to(device)
        dones  = torch.from_numpy(np.vstack([e.done for e in exp if e is not None]).astype(np.uint8)).float().to(device)#by default its bool so convert to use for arithmetic
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exp if e is not None])).float().to(device) 
        return (states,actions,rewards,next_states,dones)
                                     
    def add(self,state,action,reward,next_state,done):
        self.buffer.append(self.experience(state,action,reward,next_state,done)) 

    def __len__(self):
        return len(self.buffer)                                     
                                     
                                     
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(0.0,1.0) for i in range(len(x))])
        self.state = x + dx
        return self.state