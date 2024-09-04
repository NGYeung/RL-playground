# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:59:44 2024

@author: Yiyang Liu
"""

import random as rnd
from collections import namedtuple

# Define a tuple to store transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save aone instance of transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)