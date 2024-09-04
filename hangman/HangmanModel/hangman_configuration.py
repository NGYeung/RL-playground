# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:48:38 2024

@author: Yiyang Liu

Configuration file for hangman RL
"""



class Config:
    """ User config class """
    def __init__(self, path: str=None): 
        
        self.model = 'transformer' #or 'dqn'
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.num_episodes = 200000
        self.mem_capacity = 10000
        self.train_steps = 999999
        self.warmup = 15
        self.save_freq = 100
        self.gamma = 0.98 #for the learning rate
        
        #for exploit vs explore
        self.starteps = 1
        self.endeps = 0.05
        self.decay_eps = 200
        self.target_update_freq = 100

   
