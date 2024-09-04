# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:02:14 2024
This script is for storing multiple game agents
More games can be added, but now it's only hangman

@author: Yiyang Liu
"""



import math
import random as rnd
import numpy as np



from hangman_configuration import Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from replay_memory import ReplayMemory, Transition
import time
import logging

#https://huggingface.co/docs/transformers/en/model_doc/decision_transformer
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from Game_envs import Hangman_Env



logger = logging.getLogger('root')



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.fc2(x.squeeze(0))
        return x


class agent_hangman():
    
    def __init__(self):
        
        en = Hangman_Env(dict_file = "words_250000_train.txt")
        self.config = Config() #alpha beta gamma batch_size
        self.state_size = 27
        self.action_size = 26
        self.memory = ReplayMemory(self.config.mem_capacity)
        self.step_count = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []


        self.env = en
        self.id = int(time.time())
        
        
        #self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use Double DQN and if it's an overshoot then switch back to usual dqn
        if self.config.model == 'transformer':
            self.transformerConfig = DecisionTransformerConfig( 
            state_dim = self.state_size, act_dim = self.action_size, hidden_size = 64, action_tanh = True, 
            vocab_size = 26, n_positions = 64, n_layer = 3, n_head = 4
            ) 
            self.policy_net = DecisionTransformerModel(self.transformerConfig).to(self.device)
            self.target_net = DecisionTransformerModel(self.transformerConfig).to(self.device)
        
        if self.config.model == 'lstm':
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)
            
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = 1e-4, alpha=0.99, eps=1e-8, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.config.gamma)
        self.target_net.eval()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    
    def eps_scheduler(self):
        '''
        Adaptively balance explore or exploit, using exponential decay
        '''
        
        return self.config.endeps + (self.config.starteps - self.config.endeps)*math.exp(-1. * self.step_count / self.config.decay_eps/1000)
     


    def heuristic_action(self, state):
        guessed = state[27:]
        rand  = rnd.random()
        #print(guessed)
        letter_frequencies = 'etaoinshrdlcumwfgypbvkjxqz'
        if rand > 3/4:
            select = rnd.choice([i for i in range(26)])
            while guessed[select] == 1:
                select = rnd.choice([i for i in range(26)])
            return select
        for idx in range(26):
            char = ord(letter_frequencies[idx]) - ord('a')
            if guessed[char] == 0:
                return ord(letter_frequencies[idx]) - ord('a')
        

    
    def act(self, state, for_test = False):
        '''
        Select an action. This is the function to call when playing the game.
        '''
        
        sample = rnd.random()
        eps = self.eps_scheduler()
        if for_test:
            eps = 0.45
        self.step_count += 1
            
        if sample > eps: #exploit
            #print('exploit')
            with torch.no_grad():
                action_sequence = self.policy_net(state.float())
                #selected_action = self.policy_net(state.float()).argmax() # what if I change it to select the max unguessed letter?? --08/27
                _, actions = torch.topk(action_sequence, 26, sorted = False)
                #print(actions)
                for action_idx in actions.tolist():
                    if self.env.guessed_letters[action_idx] == 0:
                        return action_idx
                #return selected_action
        else: #explore, using heuristic
            #print('explore')
            selected_action = self.heuristic_action(state)
 
            return selected_action


    def train_n_play(self):
        '''
        The training loop
        '''
        
        #self.observation_space = Tuple((
		#	MultiDiscrete(np.array([MaxLen]*27)),  # The state space of strings 
		#	MultiBinary(26),                            # Guessed letters 0 or 1
        #    Discrete(6 + 1) # attempt left = 0-6
        #    )) 
        num_episodes = self.config.num_episodes
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        for epi_idx in range(num_episodes):
            # episode start. Initiate env.
            state = self.env.reset()
           
            
            state = torch.cat((state[0], state[1], torch.tensor(state[2]).unsqueeze(0)), dim=0)
          
            count = -1
            while True:
                count += 1
                # Select an action
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(int(action))
         
                next_state = torch.cat((next_state[0], next_state[1],torch.tensor(next_state[2]).unsqueeze(0)), dim=0)

                # Store the transition
                self.memory.push(state, action, next_state, reward)
                
                if epi_idx >= self.config.warmup: 
                    self.replay()
                    self.scheduler.step()
                    done = (count > 10) or done
                else:
                    done = (count > 10) or done
                
                    

                
                # Move to the next state
                state = next_state
                reward_in_episode += reward

                if epi_idx % 100 == 0:
                    print('episode', epi_idx)
                    self.env.render()
                    print(self.eps_scheduler())
                
                

                if done:
                    self.episode_durations.append(count + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    reward_in_episode = 0
                    break


                self.last_episode = epi_idx

                if done:
                    self.episode_durations.append(count + 1)

                    break
                
            # Update the target network
            if epi_idx % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
               
         
            if epi_idx % self.config.save_freq == 0:
                self.save(self.config.save_file_name)
    
    
    def replay(self):
        """
        essentially,
        the training loop
        """
        
        batch_size = self.config.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        
        non_empty_indicator = torch.tensor([1 if s is not None else 0 for s in batch.next_state], device=self.device, dtype=torch.bool)
        
        cat_next_states = torch.cat([s.clone().detach() for s in batch.next_state if s is not None])
        
        # collate
        cat_state = torch.cat(batch.state)
        #cat_action = np.zeros((batch_size,26))
        #for i, action in enumerate(batch.action):
        #    cat_action[i,action] = 1
        cat_action  = torch.tensor(batch.action)
        #cat_action = torch.tensor(cat_action, device=self.device, dtype=torch.int64)
        cat_reward = torch.tensor(batch.reward)
        cat_state.resize_(batch_size, 27+26+1).to(self.device).float().requires_grad_(True)
        cat_next_states.resize_(batch_size, 27+26+1).to(self.device).float().requires_grad_(True)
        
        # compute Q from policy_net
        curr_state_Q = self.policy_net(cat_state.float())[torch.arange(batch_size), cat_action] # 0827 the problem is this line.
        #print('curr Q',curr_state_Q, curr_state_Q.shape)
        
        next_state_Q = torch.zeros(batch_size, device=self.device, dtype=torch.float)

        next_state_Q[non_empty_indicator] = self.target_net(cat_next_states.float()).max(1)[0].detach()
        #print('next_Q',next_state_Q, next_state_Q.shape)
        #print('reward', cat_reward, cat_reward.shape)
        
        # Compute the expected Q values
        expected_curr_state_Q = (next_state_Q * self.config.gamma) + cat_reward

        # Try to switch betwwen Huber loss and MSE??
        if self.config.lossfn == 'huber':
            criterion = nn.SmoothL1Loss()
            loss = criterion(curr_state_Q, expected_curr_state_Q).float()

        if self.config.lossfn == 'mse':
            criterion = nn.MSELoss()
            loss = criterion(curr_state_Q, expected_curr_state_Q).float()
            
    
        self.optimizer.zero_grad()
        loss.backward()

   
        for name, param in self.policy_net.named_parameters():
 
            param.grad.data.clamp_(-1.1, 1.1)
  
        self.optimizer.step()
    
    def save(self, filename):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "reward": self.reward_in_episode,
            "episode_durations": self.episode_durations,
            "config": self.config
            }, filename)
        
    def load(self, filename = 'train_01.pth'):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy'])
        self.target_net.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_in_episode = checkpoint['reward']
        self.episode_durations = checkpoint['episode_durations']
        self.config = checkpoint['config']

    def no_train_just_play(self, tot_episodes=1000):
      

        win = 0
        lose = 0
        for episode in range(tot_episodes):
            state = self.env.reset()  # reset environment to a new, random state
            state = (state[0].reshape(-1, self.env.MaxLen), state[1]) 
            
            
            gameover = False

            while not gameover:
                action = self.act(state)
                state, reward, gameover, info = self.env.step(action)

                if reward == -10:
                    lose += 1
                        
                if reward == 10:
                    win += 1

        print(f"Results after {tot_episodes} episodes:")
        print(f"Average win_rate per episode: {win / tot_episodes}")    
        
      

        

    

class Q_hangman():
    '''
    The hangman agent.
    '''
    
    def __init__(self):
        
        en = Hangman_Env(dict_file = "words_250000_train.txt")
        self.config = Config() #alpha beta gamma batch_size
        self.state_size = 27
        self.action_size = 26
        self.memory = ReplayMemory(self.config.mem_capacity)
        self.step_count = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []

        self.env = en
        self.id = int(time.time())
        self.checkpoint = 0
        self.win = 0
        self.lose = 0
        
        # Fast and Easy
        if self.config.model == 'qtable':
            self.policy_net = defaultdict(lambda: np.zeros(26))
            
    
    def eps_scheduler(self):
        '''
        Adaptively balance explore or exploit, using exponential decay
        '''
        
        return self.config.endeps + (self.config.starteps - self.config.endeps)*math.exp(-1. * self.step_count / self.config.decay_eps/1000)
     

           
            
    def heuristic_action(self, state):
        k = len(self.env.curr_word)
        same_length = []
        for word in word_list:
            if len(word) == k:
                same_length.append(word)

        wordfreq = count_ngrams(same_length, 1)
        wordfreq = wordfreq.most_common()
        
        letter_frequencies = ''
        for key in wordfreq:
            letter_frequencies += key[0]
                
        
        guessed = state[27:]
        rand  = rnd.random()
        #print(guessed)
        #letter_frequencies = 'etaoinshrdlcumwfgypbvkjxqz'
        if rand > 2: #change this depending on DQN or just Qtable
            select = rnd.choice([i for i in range(26)])
            while guessed[select] == 1:
                select = rnd.choice([i for i in range(26)])
            return select
        for idx in range(len(letter_frequencies)):
            char = ord(letter_frequencies[idx]) - ord('a')
            if guessed[char] == 0:
                return ord(letter_frequencies[idx]) - ord('a')
        

    def correct_letter(self,state):
        correct = []
        state = state[0:27]
        for i in state:
            if 0<=i<=25:
                correct.append(chr(i+ord('a')))
        return correct


    def state2wrd(self, state):
        
        pattern = ''
        for logit in state:
            if logit == 26:
                pattern += '_'
            elif logit == -1:
                pattern += '*'
            else:
                pattern += chr(logit+ord('a'))

        return pattern
        

    def word2state(self, word):

        state = [0 for i in range(27)]
        idx = 0
        for w in word:
            if w == '_':
                state[idx] = 26
            elif word == '*':
                state[idx] = -1
            else:
                state[idx] = ord(w) -ord('a')
            idx += 1

        return torch.tensor(state)
            
                

    

    def act_stat(self, state):
        '''
        Statistical method. 1# to provide instances for RL. 2# RL may not be a good approach for this problem.
        '''

        # find length
        l = len(self.env.curr_word)

        dictionary = word_count_dict[l] #n-dictionary created outside the function.
        pattern = ''
    
        for logit in state:
            if logit == 26:
                pattern += '.'
            elif logit == -1:
                pass
            else:
                pattern += chr(logit+ord('a'))

    
        selected_dictionary = []
        for wrd in dictionary:
    
            # match pattern
            if re.match(pattern,wrd):
                selected_dictionary.append(wrd)
        

        # use this to update the n-grams
        guessed = [chr(idx + ord('a')) for idx in range(26) if self.env.guessed_letters[idx] == 1]
        correct = self.correct_letter(state)
        wrong = list(set(guessed)-set(correct))

        vowel_idx = [0,4,8,14,20]
        vowel_count = sum([1 if state[idx] in [0,4,8,14,20] else 0 for idx in range(27)])
        guess_vowel = vowel_count < 0.55 #test against 0.5 and 0.45

        
        N_gram = n_gram(selected_dictionary)
        
        unigram = N_gram.unigram.most_common()
      

        
        action = -1 #nothing is done if action = -1
        
        # return most frequently letter in possible selection of words
        for letter, freq in unigram:
            if letter not in guessed:
                code = ord(letter)-ord('a')
                if code in vowel_idx and not guess_vowel: 
                    action = code
                    continue
                action = code
                break


        # Substring dictionaries
        if action  == -1:
       
            substring_dict = self.sub_dictionary(word_list, pattern)
            l_gram = n_gram(substring_dict) 
            l_unigram = l_gram.unigram.most_common()
       
            for letter, freq in l_unigram:
                if letter not in guessed:
                    code = ord(letter)-ord('a')
                    if code in vowel_idx and not guess_vowel:
                        action = code
                        continue
                    action = code
                    break

        
        # Shorter substring.....Cascading N-gram. L-5-3-1
        if action  == -1:
       
            k = 5

            subsubstring_dict = []
            for idx in range(l-k+1):
                subpattern = pattern[idx:idx+k]
                subsubstring_dict += self.sub_dictionary(word_list, subpattern)
                
            l_gram = n_gram(subsubstring_dict) 
            l_unigram = l_gram.unigram.most_common()
        
            for letter, freq in l_unigram:
                if letter not in guessed:
                    code = ord(letter)-ord('a')
                    if code in vowel_idx and not guess_vowel:
                        action = code
                        continue
                    action = code
                    break
                    
        if action  == -1:
            k = 3
        
            subsubstring_dict = []
            for idx in range(l-k+1):
                subpattern = pattern[idx:idx+k]
                subsubstring_dict += self.sub_dictionary(word_list, subpattern)

            l_gram = n_gram(subsubstring_dict) 
            l_unigram = l_gram.unigram.most_common()
            for letter, freq in l_unigram.items():
                if letter not in guessed:
                    code = ord(letter)-ord('a')
                    if code in vowel_idx and not guess_vowel:
                        action = code
                        continue
                    action = code
                    break

        
        if action == -1:
            n_gram_act = n_gram(word_list)
            probability = n_gram_act.ngram_probability(state, weight = [1,0,0,0,0])
            sorted_act = np.argpartition(probability, -26)[-26:]
               
                #largest_k_elements = arr[largest_k_indices]
            for act in sorted_act:
                if chr(act+ord('a')) not in guessed:
                    action = act
                    
        return action
        
        
                

    def act_q(self, state, for_test = False):
        '''
        Select an action. This is the function to call when playing the game.
        For the classical reinforcement Q-table learning.
        '''
        
        sample = rnd.random()
        eps = self.eps_scheduler()
        if for_test:
            eps = 0.2
        self.step_count += 1
        countobs = sum([1 if k==26 else 0 for k in state])
        
        
        vowel_idx = [0,4,8,14,20]
        vowel = sum([1 if self.env.guessed_letters[idx]== 1 else 0 for idx in [0,4,8,14,20] ])
        
        '''
        if self.env.attempts_left  <= 2:
            #For the last two steps I'm gonna shamelessly rely on n-gram :P
            #print('ngram')
            if sample >eps:
                guessed = [chr(idx + ord('a')) for idx in range(26) if self.env.guessed_letters[idx] == 1]
                correct = self.correct_letter(state)
                wrong = list(set(guessed)-set(correct))
                n_gram_act = n_gram(word_list, list_to_exclude= wrong)
                probability = n_gram_act.ngram_probability(state[0:len(self.env.curr_word)])
                sorted_act = np.argpartition(probability, -6)[-6:]
                #largest_k_elements = arr[largest_k_indices]
                for act in sorted_act:
                    if chr(act+ord('a')) not in guessed:
                        action =  act
        '''
                
             
        if sample > eps and self.state2wrd(state) in self.policy_net: #exploit
            
            #print('exploit')
            #print(self.policy_net[self.state2wrd(state)])
            action = self.policy_net[self.state2wrd(state)].argmax()
    
            if self.env.guessed_letters[action] == 0:
                 
                return action
            else:
                temp = state.tolist() + self.env.guessed_letters
                action = self.heuristic_action(temp)
     
        elif eps > sample > 1/3*eps:
            #statistical method to avoid lack of successful update
            
            selected_action = self.act_stat(state)
            action = selected_action
        else: #explore, using heuristic
            LST = [i for i in range(26)]
            select = rnd.choice(LST)
            while self.env.guessed_letters[select] == 1:
                LST.remove(select)
                select = rnd.choice(LST)
            action = select

        return action
                

                
            
    def q_table_update(self, state, action, reward, next_state):

        state = self.state2wrd(state)
        next_state = self.state2wrd(next_state)
        
        best_next_action = np.argmax(self.policy_net[next_state])
        td_target = reward + self.config.gamma * self.policy_net[next_state][best_next_action]
        td_error = td_target - self.policy_net[state][action]
        self.policy_net[state][action] += self.config.learning_rate * td_error


    def train_q_table(self):
        '''
        for the non-nn method. no replay neede. how nice.
        '''

        num_episodes = self.config.num_episodes
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        for epi_idx in range(num_episodes):
            # episode start. Initiate env.
            state = self.env.reset()
           
            #guessed_letters = state[1]
            #attempleft = state[2]
            state = state[0]
          
            count = -1
            while True:
                count += 1
                # Select an action
                action = self.act_q(state)
          
                next_state, reward, done, _ = self.env.step(int(action))
                next_state = next_state[0]

                self.q_table_update(state, action, reward, next_state)


                done = (count > 28) or done
                
                # Move to the next state
                state = next_state
                reward_in_episode += reward

                if epi_idx % 500 == 0:
                    print('episode', epi_idx)
                    self.env.render()
                    #print(self.eps_scheduler())

                
                
                

                if done:
                    self.episode_durations.append(count + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    reward_in_episode = 0
                    break


                self.last_episode = epi_idx

                if done:
                    self.episode_durations.append(count + 1)

                    break
                    
            if epi_idx % 1000 == 0:
                    
                    self.save()
                    #print(self.eps_scheduler())
    

    def sub_dictionary(self, full_dict, word_pattern):
        '''
        create a subdictionary for high-quality prediction.
        '''
        
        full_dict
        new_dictionary = []
        l = len(word_pattern)
        for dict_word in full_dict:
            for i in range(len(dict_word)-l):
                if re.match(word_pattern,dict_word[i:i+l]):
                    new_dictionary.append(dict_word[i:i+l])
        return new_dictionary


    
    def get_Q_table(self):

        return self.policy_net


    def save(self):
        filename = 'q_table_' + str(self.checkpoint) + '.csv'
      
        #data = {key: value.tolist() for key, value in self.policy_net.items()}
        table = pd.DataFrame(self.policy_net)

        table.to_csv(filename,index=False)

        
        
        
        
        

    def load(self, checkpoint = 1):
        filename = 'q_table_' + str(checkpoint) + '.csv'
        table = pd.read_csv(filename)
 
        data = table.to_dict()
     
        self.policy_net.update(data)
        self.checkpoint = checkpoint
      

        

    
                