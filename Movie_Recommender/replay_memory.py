# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:59:44 2024

@author: Yiyang Liu
"""
import numpy as np
import numpy.random as rnd
from collections import namedtuple

# Define a tuple to store transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory_Prior:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priority = []

    def push(self, priority, prev_state, action, reward, state):
        """Save aone instance of transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priority.append(None)
        self.priority[self.position] = priority.cpu() # this is the weight in replay
        self.memory[self.position] = Transition(prev_state, action, reward, state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6):
        """Prioritize sampling: a batch of transitions."""
        
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priority)
        else:
            priorities = np.array(self.priority[:len(self.memory)])
            
        
        prob = abs(priorities) ** alpha
        #print('check1', prob, priorities)
        prob /= prob.sum()
        #prob = prob.squeeze(0)
        #print('check2', prob)
        priorities = np.array(priorities)
        priorities = np.nan_to_num(priorities, nan = 0)
        indices = np.random.choice(len(self.memory), batch_size, p=prob)
        experiences = [self.memory[i] for i in indices]
        prob = np.nan_to_num(prob, nan=0)

        # Calculate important sampling weights
        total = len(self.memory)
        weights = (total * prob[indices]) ** (-1)
        weights /= weights.max()
  

    
        return experiences, weights, indices

    def __len__(self):
        return len(self.memory)