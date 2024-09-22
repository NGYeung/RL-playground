# RL-playground-hangman-movieReco
This repository is my playground for reinforcement learning.

Project 1: Movie Recommender with (DDQN and GNN)

	- All files and training notebooks are in the Movie_Recommender folder

 	- Data Set: MovieLens 100K

 	- The Baseline model: Double DQN with STOCHASTIC PRIORITIZED REPLAY 

  		* Selected Features: movie title, movie date, genre, movie rating avg, user age, user average rating (for all and each genre), user occupation
    		* Output: Predicted rating of the movie.
      		* Metrics: MSE Loss.

  		* DQN architecture:  [FC1] -> relu -> [FC2] -> relu -> [FC3] -> log_softmax -> output

    		

    		

	

Project 2: Q-learning for Hangman.

	- All files and training notebooks are in the hangman folder.

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

  	- Result after one epoch on table-based Q: 61%




