# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:03:55 2024

@author: Yiyang Liu
"""


class Config:
    """ User config class """
    def __init__(self, path: str=None): 
        
        self.model = 'base' #or 'transformer'
        self.batch_size = 64
        self.learning_rate = 1e-2
        self.num_episodes = 200000
        self.mem_capacity = 5000
        self.warmup = 100
        self.gamma = 0.95 #for the learning rate
        
        #for exploit vs explore
        self.starteps = 1
        self.endeps = 0.05
        self.decay_eps = 200
        self.state_size = 64
        
        self.render_freq = 1000
        self.target_update_freq = 100
        self.save_freq = 1000
        self.save_file_name = r'C:\Users\yyBee\RL\Model_checkpoints\base_dqn_movie.pth'



class Config4Colab:
    """ User config class """
    def __init__(self, path: str=None): 
        
        self.model = 'base' #or 'transformer'
        self.batch_size = 64
        self.learning_rate = 1e-2
        self.num_episodes = 200000
        self.mem_capacity = 5000
        self.warmup = 100
        self.gamma = 0.95 #for the learning rate
        
        #for exploit vs explore
        self.starteps = 1
        self.endeps = 0.05
        self.decay_eps = 200
        self.state_size = 64
        
        self.render_freq = 1000
        self.target_update_freq = 100
        self.save_freq = 1000
        self.save_file_name = r'drive/'
