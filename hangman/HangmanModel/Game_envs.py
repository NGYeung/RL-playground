# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game environments.
More games can be added.

@author: Yiyang Liu
"""
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, MultiBinary, Tuple
import numpy as np
import random as rnd
import logging


logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class Hangman_Env(gym.Env):
    """Custom Hangman game Environment that follows gym interface"""
    
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dict_file = None, MaxLen = 27):
        """Initialize the hangman game environment. """
        
        
        super(Hangman_Env, self).__init__()
       
        self.action_space = Discrete(26)  # the alphabet a-z = ord(x)-ord('a')
        
        self.observation_space = Tuple((
			MultiDiscrete(np.array([27]*MaxLen)),  # The state space of strings  
			MultiBinary([2]*26),# Guessed letters a-z guessed 1 unguessed 0
            Discrete(6 + 1) # attempt left = 0-6
            )) 
        #Example
        #observation 0: a___e = [0 26 26 26 4 0000000..]
        #observation 1:  only a and e are guessed [1 0 0 0 1 0000]
        #observation 2: 3 attemps are left [3]
        
        if dict_file is None:
            self.dictionary = list('no training dictionary for this game')
        else:
            f = open(dict_file, 'r').readlines()
            self.dictionary = [word.strip() for word in f]
        
        self.curr_word = ""
        self.unknown_word = []
        self.guessed_letters = np.zeros(26, dtype=int)
        self.attempts_left = 6
        self.win = False
        self.MaxLen = MaxLen
            
        self.reset()

    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Choose a random word from the list
        self.curr_word = rnd.choice(self.dictionary) #this need to change when attaching to the api
        self.unknown_word = ['_'] * len(self.curr_word)  # Hidden word representation
        self.guessed_letters = [0] * 26  # Initialize as all unguessed
        self.attempts_left = 6  # 6 guesses for each word
        logger.info("Reset: new word! new round!")
        logger.info("Reset: New word is [" + self.curr_word +"]")
        
        return np.array(self.guessed_letters)
    
    def step(self, action):
        
       assert self.action_space.contains(action), f"Invalid Action: {action} is not in the action space"
       # Just to help with debugging
       
       done = False
       reward = 0

       # Convert action to character
       char2guess = chr(action + ord('a'))

       # If the character was already guessed, return current state with no reward
       if self.guessed_letters[action] == 1:
           return self._get_observation(), reward, done, {}

       self.guessed_letters[action] = 1

       if char2guess in self.curr_word:
           # Correct guess
           reward = 1
           for idx, char in enumerate(self.curr_word):
               if char == char2guess:
                   self.unknown_word[idx] = char2guess
           if '_' not in self.unknown_word:
               done = True
               reward = 10  # Extra reward for winning
       else:
           # Incorrect guess
           self.attempts_left -= 1
           if self.attempts_left == 0:
               done = True
               reward = -10  # Penalty for losing

       return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        
        unknown_word_logits = [26 if char == '_' else ord(char)-ord('a') for char in self.unknown_word]
        padding = [0]*(self.MaxLen-len(self.unknown_word_logits))
        # 1 =  know 0 = unknown
        return unknown_word_logits+padding, self.guessed_letters, self.attempts_left, {}
    
    
    def render(self, mode='human'):
        unknown_word_str = " ".join(self.unknown_word)
        guessed_letters_str = ", ".join([chr(i + ord('a')) for i, val in enumerate(self.guessed_letters) if val == 1])
        print(f"Word: {unknown_word_str}")
        print(f"Guessed Letters: {guessed_letters_str}")
        print(f"Attempts Left: {self.attempts_left}")
        
    

    