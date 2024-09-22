# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game environments.
More games can be added.

@author: Yiyang Liu
"""
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiBinary, Tuple
import numpy as np
import random as rnd
import logging
from typing import List
from numba import njit

from MovieLens import Movie_100K, Data2State


logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class Reco_Env(gym.Env):
    """Baseline Hangman game Environment that follows gym interface
    Game: Given observations of movies 
    """
    
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, ini_dataset = None, embedding_size = 64):
        """Initialize the hangman game environment. """
        
        
        super(Reco_Env, self).__init__()
       
        self.action_space = Discrete(5)  # action i correspondent to score i+1
       
        
        self.observation_space = Tuple(Box(low=1.0, high=5.0, shape = (2,)),
                                       Box(low = -1, high = 1, shape = (8,)),
                                       Box(low=-1, high = 3,shape=(5,)),
                                       MultiBinary(19),
                                       Discrete(99),
                                       Discrete(2),
                                       MultiBinary(20)
                                       )
        # An observation includes:
        # observation 0: the average rating of the film and the average rating given by the user
        # observation 1: title_embedding. Choose embedding length = 8
        # observation 2: move date 
        # observation 3: movie genre. = MultiDiscrete([2]*19)
        # observation 4: user age
        # observation 5: user gender
        # observation 6: user occupation
        # observation 7: user location we excluded the location so far
             
        self.database = ini_dataset
        self.state = None
        self.rating = -1 # the ground truth
        self.action = -1
        self.reward = 0


    def reset(self):
        """Reset the state of the environment to an initial state"""
        idx = rnd.randint(0, 100000)
        data_item = self.database[idx]
        self.state = Data2State(data_item)
        self.rating = data_item['rating']
    
    def step(self, action: int) -> int:
        '''
        take the action chosen by agent as the input, give reward and update the state.
        
        Current game rules: (used as the baseline)
            if action == rating: +3, 
            if abs(action+1 - rating) == 1: +1
            if abs(action+1 - rating) == 2: -1
            if abs(action+1 - rating >= 3): -3
        '''
        
        self.action = action
        
        measure = abs(action - self.rating)
        reward = 0
        
        if measure == 0:
            reward = 3
        elif measure == 1:
            reward = 1
        elif measure == 2:
            reward = -1
        else: 
            reward = -3
        
        return reward
        
    
    
    def render(self, mode='human'):
        
        print(f"Movie: {self.data_item['title']}")
        print(f"User Rating: {self.rating}")
        print(f"Predicted Rating: {self.action}")
        print(f"Reward: {self.reward}")
        
    
    
    

    