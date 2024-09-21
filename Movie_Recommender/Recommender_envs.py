# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game environments.
More games can be added.

@author: Yiyang Liu
"""
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Discrete, MultiBinary, Tuple
import numpy as np
import random as rnd
import logging


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
                                       Discrete(2023),
                                       MultiBinary(19),
                                       Discrete(99),
                                       Discrete(2),
                                       MultiBinary(21),
                                       Discrete(11)
                                       ) 
        # An observation includes:
        # observation 0: the average rating of the film and the average rating given by the user
        # observation 1: title_embedding. Choose embedding length = 8
        # observation 2: move date 
        # observation 3: movie genre. = MultiDiscrete([2]*19)
        # observation 4: user age
        # observation 5: user gender
        # observation 6: user occupation
        # observation 7: user location
             
        self.database = ini_dataset
        self.state = (np.zeros(2,),np,zeros(8,))


    def new_game(self):
        """Reset the state of the environment to an initial state"""
        idx = rnd.randint(0, 100000)
        data_item = self.database[idx]
        
        self.observations['film_avg']  = data_item['film_avg']
        self.observations['user_avg']  = data_item['user_avg']
        self.observations['film_avg']  = data_item['film_avg']
        self.observations['film_avg']  = data_item['film_avg']
    
    def step(self, action):
        '''
        take the action chosen by agent as the input, give reward and update the state.
        '''
    
        
        # initialize reward
        
        
    
    def translate(self, state_vec):
        '''
        From state vector to Data for rendering
        '''
       
        
       
        
    def _get_observation(self):
        '''
        Return the current state in a dictionary to calculate rewards
        film_avg: float - average rating of the film
        user_avg: float - average rating given by the user
        '''
        
        observations = {}
        observations['film_avg'] = state
        observations['title_emb'] = 
        
        pass
    
    def render(self, mode='human'):
        
        
        
        pass
    

    