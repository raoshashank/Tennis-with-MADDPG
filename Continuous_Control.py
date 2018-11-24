from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle
env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher_Linux_NoVis/Reacher.x86_64') 
import matplotlib.pyplot as plt

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print("~~~~~~~~~Here~~~~~~")
print(brain_name)
# In[4]:


env_info = env.reset(train_mode=True)[brain_name]
env.reset(train_mode=True)[brain_name]
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

import importlib
import Agent_1
importlib.reload(Agent_1)


# In[7]:


from Agent_1 import Agent
agent = Agent(state_size = state_size, action_size = action_size,seed = 2,num_agents=num_agents)

#agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
#agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

# In[8]:


from collections import deque
import pickle
import sys
pass_score = 30
import torch
def ddpg(n_episodes = 10000, t_max = 1000, print_every = 100,num_agents=20):
    scores_deque = deque(maxlen=100)
    scores=[]
    count=0
    mean_score = 0 
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        
        agent.reset()
        agent_scores=np.zeros(num_agents)
        
        for t in range(t_max):
            actions=[]
            actions = agent.act(states)
            env_info_ = env.step(actions)[brain_name]
            next_states = env_info_.vector_observations
            rewards = env_info_.rewards
            dones = env_info_.local_done 
            #add all the experience to common replay buffer (Attempt #1)
            for i in range(0,num_agents):
                agent.add_experience(states[i],actions[i],rewards[i],next_states[i],dones[i])
            #Every 20 time steps, perform 10 learning steps (Attempt #3)
            agent.step(t)
            states=next_states
            agent_scores+=rewards
            if np.any(dones):
                break
                
        scores_deque.append(np.mean(agent_scores))
        scores.append(np.mean(agent_scores))
        mean_score = np.mean(scores_deque)
        sys.stdout.write("\r %i %f" % (int(i_episode),float(mean_score)))
        if i_episode %10 == 0:
            print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            torch.save(agent.actor_local.state_dict(),  'checkpoint_actor_1.pth' )
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_1.pth')
            with open('scores','wb') as fp:
                pickle.dump(scores,fp)
            fp.close()
            
         
        if mean_score >=30.0 and count == 0: #and suc_ep>=100:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        elif mean_score >=30.0 and count>0:
                count+=1
                if count >= 100:
                    torch.save(agent.actor_local.state_dict(),  'checkpoint_actor_1.pth' )
                    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_1.pth')
                    with open('scores','wb') as fp:
                        pickle.dump(scores,fp)
                    fp.close()
                    break

    return scores

scores = ddpg(num_agents = num_agents)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[12]:


env.close()

