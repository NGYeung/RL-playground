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

from transformers import BertTokenizer
#import spacy




filepath = r"C:\Users\yyBee\Datasets\ml-100k\movie_100k_all.csv"

class Movie_100K():
    
    def __init__(self, filename = filepath, for_training = False):
        '''
        Input: filename[dict] = {rating_file:path, user_file:path, movie_file:path}

        '''
        
        self.encode_title = 0 
        self.item = {}
        self.path = filename
        self.load()
        self.text_tokenizer = None
        if for_training:
            self.encode_title = 1
            #self.text_tokenizer = BertTokenizer('' )

        
        
    def __len__(self):
        
        return self.rating.shape[0]
    
        
    
    def __getitem__(self, idx):
        '''
        Input: the index
        return items: dictionary
        {'timestamp':[Int] the time-stamp of the recommendation, 
         'user':[Dict] a dictionary of user profile,
         'movie':[Dict] Information of the rated movie. }
        
        movie:{'movieid':[Int], 'title':[Str], 'date':[Int], 'genre':nparray }
        user:{'user_id':[Int], 'age':[Int], gender:[Int] 'occupation':nparray}
        '''
        self.item = self.data.iloc[idx,2:].to_dict()
        
        #orient='records' exclude the index
        return self.item
    
    
    def load(self):
        '''
        Load Data and Process
        1. convert the occupation to one-hot
        2. DateTime to timestamp for the movie date
        3. Gender to boolean.
        '''
        self.data = pd.read_csv(self.path)
        self.data = self.data.apply(self.encode_user, axis = 1)
        self.data = self.data.apply(self.encoder_dates, axis = 1)


        
        
    def encode_user(self,row):
        '''
        convert occupation to vector
        '''
        # process the occupation
        list_of_jobs = {'student':0, 'other':19, 'educator':1, 'administrator':2,'engineer':3,
                        'programmer':4, 'librarian':5, 'writer':6, 'executive':7, 'scientist':8,
                        'artist':9, 'technician':10, 'marketing':11, 'entertainment':12, 'healthcare':13, 
                        'retired':14,'lawyer':15,'salesman':16,'homemaker':17, 'doctor':18}
        out = np.zeros((20,))
        key = row['occupation'] 
        if key in list_of_jobs:
            out[list_of_jobs[key]] = 1
        row['occupation'] = out
        
        if row['gender'] == "M":
            row['gender'] = 1
        else:
            row['gender'] = 0 
        
        return row
    
    
    
    def encoder_dates(self,row):
        '''
        Encode the dates of the movie
        input format: Jan-01-1995
        '''
        
        date = pd.to_datetime(row['date'])
        
        year = date.year
        mon = date.month
        day = date.day
        #print(year, mon, day)
        dt_vec = []
        dt_vec.append((2000-year)/5)
        #month encoding
        dt_vec += [np.sin(2 * np.pi * mon/ 12), np.cos(2 * np.pi * mon / 12)]
        dt_vec += [np.sin(2 * np.pi * day/ 12), np.cos(2 * np.pi * day / 31)]
        
        row['date'] = np.array(dt_vec)
        return row
    

    

    
        
        
    
        
        
        
        