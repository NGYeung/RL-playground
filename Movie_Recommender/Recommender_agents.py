# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game agents
More games can be added, but now it's only hangman

@author: Yiyang Liu
"""



import math
import random as rnd
import numpy as np



from Reco_config import Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from replay_memory import ReplayMemory_Prior, Transition
import time
import logging

#https://huggingface.co/docs/transformers/en/model_doc/decision_transformer
#from transformers import DecisionTransformerConfig, DecisionTransformerModel
from Recommender_envs import Reco_Env



logger = logging.getLogger('root')



class base_DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(base_DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Reco_Agent():
    '''
    A baseline agent for RL recommender
    '''
    
    def __init__(self, dataset):
        
        en = Reco_Env()
        self.config = Config() #alpha beta gamma batch_size
        self.state_size = Config.self.state_size
        self.action_size = 5
        self.memory = ReplayMemory_Prior(self.config.mem_capacity)

        self.env = en
        self.id = int(time.time())
        
        #self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use Double DQN and if it's an overshoot then switch back to usual dqn
        if self.config.model == 'base':
            self.policy_net = base_DQN(self.state_size, self.action_size)
            self.target_net = base_DQN().to(self.device)
            
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = 1e-4, alpha=0.99, eps=1e-8, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.config.gamma)
        self.target_net.eval()
    
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    
    def eps_scheduler(self):
        '''
        Adaptively balance explore or exploit, using exponential decay
        '''
        return self.config.endeps + (self.config.starteps - self.config.endeps)*math.exp(-1. * self.step_count / self.config.decay_eps/1000)
             

    
    def act(self, state, for_test = False):
        '''
        Select an action. This is the function to call when playing the game.
        '''
        
        sample = rnd.random()
        eps = self.eps_scheduler()
            
        if sample > eps: #exploit
            #print('exploit')
            with torch.no_grad():
                action_sequence = self.policy_net(state.float())
                #selected_action = self.policy_net(state.float()).argmax() # what if I change it to select the max unguessed letter?? --08/27
                _, actions = torch.topk(action_sequence, 26, sorted = False)
                #print(actions)
                for action_idx in actions.tolist():
                    if self.env.guessed_letters[action_idx] == 0:
                        return action_idx
                
        else: 
            
            selected_action = rnd.choice([0,1,2,3,4])
            return selected_action


    def train_n_play(self):
        '''
        The training loop
        '''
        
        #self.observation_space = Tuple((
		#	MultiDiscrete(np.array([MaxLen]*27)),  # The state space of strings 
		#	MultiBinary(26),                            # Guessed letters 0 or 1
        #    Discrete(6 + 1) # attempt left = 0-6
        #    )) 
        num_episodes = self.config.num_episodes
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        for epi_idx in range(num_episodes):
            # episode start. Initiate env.
            state = self.env.reset()
           
            
            state = torch.cat((state[0], state[1], torch.tensor(state[2]).unsqueeze(0)), dim=0)
          
            count = -1
            while True:
                count += 1
                # Select an action
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(int(action))
         
                next_state = torch.cat((next_state[0], next_state[1],torch.tensor(next_state[2]).unsqueeze(0)), dim=0)

                # Store the transition
                self.memory.push(state, action, next_state, reward)
                
                if epi_idx >= self.config.warmup: 
                    self.replay()
                    self.scheduler.step()
                    done = (count > 10) or done
                else:
                    done = (count > 10) or done
                
                    

                
                # Move to the next state
                state = next_state
                reward_in_episode += reward

                if epi_idx % 100 == 0:
                    print('episode', epi_idx)
                    self.env.render()
                    print(self.eps_scheduler())
                
                

                if done:
                    self.episode_durations.append(count + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    reward_in_episode = 0
                    break


                self.last_episode = epi_idx

                if done:
                    self.episode_durations.append(count + 1)

                    break
                
            # Update the target network
            if epi_idx % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
               
         
            if epi_idx % self.config.save_freq == 0:
                self.save(self.config.save_file_name)
    
    
    def replay(self):
        """
        essentially,
        the training loop
        """
        
        batch_size = self.config.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        
        non_empty_indicator = torch.tensor([1 if s is not None else 0 for s in batch.next_state], device=self.device, dtype=torch.bool)
        
        cat_next_states = torch.cat([s.clone().detach() for s in batch.next_state if s is not None])
        
        # collate
        cat_state = torch.cat(batch.state)
        #cat_action = np.zeros((batch_size,26))
        #for i, action in enumerate(batch.action):
        #    cat_action[i,action] = 1
        cat_action  = torch.tensor(batch.action)
        #cat_action = torch.tensor(cat_action, device=self.device, dtype=torch.int64)
        cat_reward = torch.tensor(batch.reward)
        cat_state.resize_(batch_size, 27+26+1).to(self.device).float().requires_grad_(True)
        cat_next_states.resize_(batch_size, 27+26+1).to(self.device).float().requires_grad_(True)
        
        # compute Q from policy_net
        curr_state_Q = self.policy_net(cat_state.float())[torch.arange(batch_size), cat_action] # 0827 the problem is this line.
        #print('curr Q',curr_state_Q, curr_state_Q.shape)
        
        next_state_Q = torch.zeros(batch_size, device=self.device, dtype=torch.float)

        next_state_Q[non_empty_indicator] = self.target_net(cat_next_states.float()).max(1)[0].detach()
        #print('next_Q',next_state_Q, next_state_Q.shape)
        #print('reward', cat_reward, cat_reward.shape)
        
        # Compute the expected Q values
        expected_curr_state_Q = (next_state_Q * self.config.gamma) + cat_reward

        # Try to switch betwwen Huber loss and MSE??
        if self.config.lossfn == 'huber':
            criterion = nn.SmoothL1Loss()
            loss = criterion(curr_state_Q, expected_curr_state_Q).float()

        if self.config.lossfn == 'mse':
            criterion = nn.MSELoss()
            loss = criterion(curr_state_Q, expected_curr_state_Q).float()
            
    
        self.optimizer.zero_grad()
        loss.backward()

   
        for name, param in self.policy_net.named_parameters():
 
            param.grad.data.clamp_(-1.1, 1.1)
  
        self.optimizer.step()
    
    
    def save(self, filename):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "reward": self.reward_in_episode,
            "config": self.config
            }, filename)
        
    def load(self, filename = 'train_01.pth'):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy'])
        self.target_net.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_in_episode = checkpoint['reward']
        self.config = checkpoint['config']


    def no_train_just_play(self, tot_episodes=1000):
      

        win = 0
        lose = 0
        for episode in range(tot_episodes):
            state = self.env.reset()  # reset environment to a new, random state
            state = (state[0].reshape(-1, self.env.MaxLen), state[1]) 
            
            
            gameover = False

            while not gameover:
                action = self.act(state)
                state, reward, gameover, info = self.env.step(action)

                if reward == -10:
                    lose += 1
                        
                if reward == 10:
                    win += 1

        print(f"Results after {tot_episodes} episodes:")
        print(f"Average win_rate per episode: {win / tot_episodes}")    
        
      

        

    



    
                