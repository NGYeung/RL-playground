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



from Reco_config import Config4Colab
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from replay_memory import ReplayMemory_Prior
import time
import logging

from Recommender_envs import Reco_Env



logger = logging.getLogger('root')



class base_DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(base_DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        #self.softmax = F.log_softmax(action_size)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)
        x = F.log_softmax(x,dim=1)
        #print(x.shape)
        return x


class Reco_Agent():
    '''
    A baseline agent for RL recommender
    '''
    
    def __init__(self, train_data, test_data):
        
        en = Reco_Env(train_data, test_data)
        self.config = Config4Colab() #alpha beta gamma batch_size
        self.state_size = self.config.state_size
        self.action_size = 5
        self.memory = ReplayMemory_Prior(self.config.mem_capacity)
        self.step_count = 0
        self.reward_in_episode = 0
        self.env = en
        self.id = int(time.time())
        
        #self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'nums GPU: {torch.cuda.device_count()}')
        
        # Use Double DQN and if it's an overshoot then switch back to usual dqn
        if self.config.model == 'base':
            print("\n \t ----------- Model = Baseline Double DQN ------------")
            print("\t Loading Policy Net ......")
            self.policy_net = base_DQN(self.state_size, self.action_size).to(self.device)
            print("\t *Total Params* = ",sum(p.numel() for p in self.policy_net.parameters()))
            print("\t *Trainable Params* = ",sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad))
            
            print("\t Loading Target Net ......")
            self.target_net = base_DQN(self.state_size, self.action_size).to(self.device)
            
      
            
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = 1e-4, alpha=0.99, eps=1e-8, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.config.gamma)
        self.target_net.eval()
        print("\n \t ----------- Model Loaded ------------")
    
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    
    def eps_scheduler(self):
        '''
        Adaptively balance explore or exploit, using exponential decay
        '''
        return self.config.endeps + (self.config.starteps - self.config.endeps)*math.exp(-1. * self.step_count / self.config.decay_eps/50)
             

    
    def act(self, state):
        '''
        Select an action. This is the function to call when playing the game.
        '''
        self.step_count += 1
        sample = rnd.random()
        eps = self.eps_scheduler()
        Q_sequence = self.policy_net(state.unsqueeze(0))
        #print('sequence:', Q_sequence)
            
        if sample > eps: #exploit
            with torch.no_grad():
              
                Q, action = torch.topk(Q_sequence, 1, sorted = False)
                #print('policy: Q, action', Q, action)
                Q = Q.squeeze().squeeze()
                action = action.squeeze().squeeze()
              
                
        else: 
            
            # explore by randomly rate the film and observe the reward
            action = rnd.choice([0,1,2,3,4])
            Q = Q_sequence.squeeze().detach()[action]
            #print('random: Q, action', Q, action)
            
        return int(action), Q
        
        

    def train(self):
        '''
        The training/playing loop
        '''
        
        start = time.time()

        num_episodes = self.config.num_episodes
        
        mseloss = 0
      
        state, rating = self.env.reset()
        state =  state.to(self.device)
        rating =  rating.to(self.device)
        for epi_idx in range(num_episodes):
            # episode start. Initiate env.
            
            
            action, Q = self.act(state)
        
            reward = self.env.step(int(action))
            self.reward_in_episode = reward
            
            prev_state = state
            
            mseloss += (rating - action -1)**2
            
            if epi_idx % self.config.render_freq == 0:
                print('episode:', epi_idx)
                self.env.render()
                print('exploit vs explore:', self.eps_scheduler())
            
            state, rating = self.env.reset()
            state =  state.to(self.device)
            rating =  rating.to(self.device)
            
            # Compute the TD error for prioritized experience replay
            Q_next = torch.max(self.policy_net(state.unsqueeze(0)))

            TD_error = reward + self.config.gamma* Q_next - Q
            
            if not TD_error:
                
                print('ErrorShape',TD_error.shape, Q_next, Q, reward, self.config.gamma)
                TD_error = 0
            
            
            # store the transition
            self.memory.push(TD_error.detach(), prev_state, action, reward, state)
            
                
            if epi_idx >= self.config.warmup: 
                self.replay()
                self.scheduler.step()
   
            # render frequency
            
                
        
            # Update the target network
            if epi_idx % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
               
         
            if epi_idx % self.config.save_freq == 0:
                self.save(self.config.save_file_name)
                
        mseloss = mseloss/num_episodes
        
        print(f"1 Training epoch {num_episodes} done, the average training loss is {mseloss}")
        print(f"Total time for training: {time.time() -  start} seconds")
        
                
                
                
    def test(self):
        
        '''Evaluate the trained model on the testing dataset.'''
        go = True
        start = time.time()
        Error = 0
        counter = 0
        while go:
            counter += 1
            state, rating, go = self.env.reset_for_eval()
            state =  state.to(self.device)
            rating =  rating.to(self.device)
            action, _ = self.act(state)
            
            Error += (action - rating - 1) **2
        
        Error = Error/counter
        
        print("Test run finished. The final testing loss is: {Error}")
        print(f"Total time for evaluation: {time.time() -  start} seconds")
        

            
            
    
    
    def replay(self):
        """
        essentially,
        the training loop
        """
        
        batch_size = self.config.batch_size
        if len(self.memory) < batch_size:
            return
        
        # Stochastic Prioritized Sampling
        experiences, weights, indices = self.memory.sample(batch_size)
        
        states = [i.state for i in experiences]
        next_states = [i.next_state for i in experiences]
        rewards = [i.reward for i in experiences]
        actions = [i.action for i in experiences]

        non_empty_indicator = torch.tensor([1 if s is not None else 0 for s in next_states], device=self.device, dtype=torch.bool)

        # All experiences to tensor
        weights = torch.tensor(weights).to(self.device)
        cat_next_states = torch.cat(next_states)
        cat_state = torch.cat(states)
        cat_action  = torch.tensor(actions).to(self.device)
        cat_reward = torch.tensor(rewards).to(self.device)
        cat_state.resize_(batch_size, self.state_size).to(self.device).float().requires_grad_(True)
        cat_next_states.resize_(batch_size, self.state_size).to(self.device).float().requires_grad_(True)
        
        
        # compute Q from policy_net
        curr_state_Q = self.policy_net(cat_state.float())[torch.arange(batch_size), cat_action] 
        next_state_Q = torch.zeros(batch_size, device=self.device, dtype=torch.float)
        next_state_Q[non_empty_indicator] = self.target_net(cat_next_states.float()).max(1)[0].detach()
    
        # Compute the expected Q values
        expected_curr_state_Q = (next_state_Q * self.config.gamma) + cat_reward
        


        # Compute the weighted MSELoss
        loss = (curr_state_Q - expected_curr_state_Q)**2 
        loss = loss * weights
        loss = loss.mean()
            
        # reset the gradiant
        self.optimizer.zero_grad()
        loss.backward()

   
        for name, param in self.policy_net.named_parameters():
            
            pass
            
            #Gradient clipping. But let's first see how well it works when it doesn't involve.
            #param.grad.data.clamp_(-1.1, 1.1)
  
        self.optimizer.step()
    
    
    def save(self, filename):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "reward": self.reward_in_episode,
            "config": self.config
            }, filename)
        
    def load(self, filename = '\content\drive\MyDrive\RL\Movie100K.pt'):
        checkpoint = torch.load(filename)
    
        # Check for incompatible keys
        result = self.policy_net.load_state_dict(checkpoint['policy'], strict=False)
        if result.missing_keys:
            print(f"Missing keys in policy: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys in policy: {result.unexpected_keys}")        
        self.policy_net.to(self.device)
    
        result = self.target_net.load_state_dict(checkpoint['target'], strict=False)
        if result.missing_keys:
            print(f"Missing keys in target: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys in target: {result.unexpected_keys}")    
        self.target_net.to(self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_in_episode = checkpoint['reward'] 
        self.config = checkpoint['config']
    


      

        

    



    
                