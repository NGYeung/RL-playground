# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game agents
More games can be added, but now it's only hangman

@author: Yiyang Liu
"""

from re import T
import gymnasium as gym
import math
import random as rnd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import PIL

from torch.cuda import init
from torch.autograd import Variable
from config import Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory
from log import setup_custom_logger
import time
import yaml
import logging

#https://huggingface.co/docs/transformers/en/model_doc/decision_transformer
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

from Game_envs import Hangman_Env

'''
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

'''


# create logger
#logger = logging.getLogger('root')

#config = None

class agent_hangman():
    
    def __init__(self, state_size, action_size, config ):
        
        en = Hangman_Env()
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = None
        self.done = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []

        self.env = en
        self.id = int(time.time())
        self.config = config #alpha beta gamma batch size
        self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # need function to generate all the training arguments
        self.policy_net = DecisionTransformerModel().to(self.device)
        self.target_net = DecisionTransformerModel().to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def sample_memory(self):
        indices = np.random.choice(len(self.memory), self.batch_size)
        batch = [self.memory[idx] for idx in indices]
        return batch

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        
        







