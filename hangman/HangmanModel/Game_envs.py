# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game environments.
More games can be added.

@author: Yiyang Liu
"""
import gymnasium as gym
from gymnasium.spaces import Text, MultiDiscrete, Discrete, MultiBinary, Tuple
import numpy as np
import random as rnd
import yaml
import logging



conf_file = None #input this later



with open(conf_file, 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class Hangman_Env(gym.Env):
    """Custom Hangman game Environment that follows gym interface"""
    
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dict_file = None, MaxLen = 29):
        """Initialize the hangman game environment. """
        
        
        super(Hangman_Env, self).__init__()
       
        self.action_space = Discrete(26)  # the alphabet a-z
        
        self.observation_space = Tuple((
			MultiDiscrete(np.array([MaxLen]*27)),  # The state space of strings 
			MultiBinary(26),                            # Guessed letters 0 or 1
            Discrete(6 + 1) # attempt left = 0-6
            ))   
        
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
        self.guessed_letters = [-1] * 26  # Initialize as all unguessed
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
        unknown_word_logits = [0 if char == '_' else 1 for char in self.unknown_word]
        # 1 =  know 0 = unknown
        return unknown_word_logits, self.guessed_letters, self.attempts_left
    
    
    def render(self, mode='human'):
        unknown_word_str = " ".join(self.unknown_word)
        guessed_letters_str = ", ".join([chr(i + ord('a')) for i, val in enumerate(self.guessed_letters) if val == 1])
        print(f"Word: {unknown_word_str}")
        print(f"Guessed Letters: {guessed_letters_str}")
        print(f"Attempts Left: {self.attempts_left}")
        
    def word2logit(self, word):
        
        word = word.strip().lower()
        if len(word) < 3:
            return None, None, None

        logits = np.zeros((len(word), 1))

		# k = 01234...26 a-z_
        char_hash = {k: [] for k in range(27)}

        for i, c in enumerate(word):
            idx = ord(c)-ord('a')
            if 0 > idx and idx > 25:
                idx = 26
			#update chars dict
            char_hash[idx].append(i)
			#one-hot encode
            logits[i][0] = idx


        return logits, char_hash
        

    def close(self):
        # Clean up if necessary
        pass
        
    