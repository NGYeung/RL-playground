# RL-playground-hangman-movieReco
This repository is my playground for reinforcement learning.

Project 1: Q-learning for Hangman.

	- This repository provided gym environments and agents for both table-based and Deep NN Q-learning. 

	- State: 
		observation 0: the state of current word 0-25 <=> a-z, and 26 = "_".
		observation 1: guessed letters. Binary(26). 1:guessed 0: not guessed
		observation 2: attempts left - 0 to 6

	- Action: choose between a-z. encoded as 0-25

	- Reward Rule: 
		correct guess +3
		wrong guess -1
		win game + 10
		lose game -10



Project 2: Movie Reco
