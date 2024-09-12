# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:20:16 2024

@author: Yiyang Liu

The Dataset Class for the class, to help with the training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Discrete, MultiBinary, Tuple




filepath = {}
filepath['movie_file'] = r"C:\Users\yyBee\Datasets\ml-100k\movies_info.csv"
filepath['ratings_file'] = r"C:\Users\yyBee\Datasets\ml-100k\rating_info.csv"
filepath['users_file'] = r"C:\Users\yyBee\Datasets\ml-100k\user_info.csv"


class Movie_100K():
    
    def __init__(self, filename = filepath):
        '''
        Input: filename[dict] = {rating_file:path, user_file:path, movie_file:path}

        '''
        
        self.item = {}
        self.path = filename
        self.load()
    
        
    
    def __get__(self, idx):
        '''
        Input: the index
        return items: 
            
        '''
        item = self.rating.iloc[idx]
        
        self.item['rating'] = item['rating']
        userid = item['user_id']
        movieid = item['item_id']
        self.item['timestamp'] = item['timestamp']
        self.item['movie'] = self.movies.iloc[movieid-1].todict()
        self.item['user'] = self.users.iloc[userid-1].todict()

        return self.item
    
    
    def load(self):
        '''
        Load Data
        '''
        self.rating = pd.read_csv(self.path['rating_file'])
        self.users = pd.read_csv(self.path['users_file'])
        self.movies = pd.read_csv(self.path['movie_file'])
        
        