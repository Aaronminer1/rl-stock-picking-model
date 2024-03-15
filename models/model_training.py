# models/model_training.py

import numpy as np
from tqdm import tqdm
from rl_model import RLModel
from data_preprocessing import preprocess_data
from config import CONFIG

class ModelTrainer:
    def __init__(self, data_file, model_file):
        self.data_file = data_file
        self.model_file = model_file
        self.rl_model = None
        
        self.state_dim = CONFIG['state_dim']
        self.action_dim = CONFIG['action_dim']
        self.learning_rate = CONFIG['learning_rate']
        self.batch_size = CONFIG['batch_size']
        self.num_episodes = CONFIG['num_episodes']
        self.max_steps = CONFIG['max_steps']
        
    def preprocess_data(self):
        """
        Preprocess the training data.
        """
        self.data = preprocess_data(self.data_file)
        
    def initialize_model(self):
        """
        Initialize the reinforcement learning model.
        """
        self.rl_model = RLModel(self.state_dim, self.action_dim, self.learning_rate)
        
    def train_model(self):
        """
        Train the reinforcement learning model.
        """
        for episode in tqdm(range(self.num_episodes), desc="Training Episodes"):
            state = self.data.iloc[0].values
            total_reward = 0
            
            for step in range(self.max_steps):
                action = self.rl_model.get_action(state)
                next_state, reward, done = self.get_next_state_reward(state, action)
                
                self.rl_model.train_step(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        
        self.rl_model.save_model(self.model_file)
        
    def get_next_state_reward(self, state, action):
        """
        Get the next state and reward based on the current state and action.
        
        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            
        Returns:
            tuple: A tuple containing the next state, reward, and done flag.
        """
        next_state = self.data.iloc[self.data.index.get_loc(state.name) + 1].values
        reward = self.calculate_reward(state, action, next_state)
        done = self.data.index[-1] == next_state.name
        
        return next_state, reward, done
    
    def calculate_reward(self, state, action, next_state):
        """
        Calculate the reward based on the current state, action, and next state.
        
        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            next_state (np.ndarray): The next state.
            
        Returns:
            float: The calculated reward.
        """
        # Implement your reward calculation logic here
        reward = 0.0
        
        # Example reward calculation based on price change
        price_change = next_state[-1] - state[-1]
        if action == 0:  # Buy
            reward = price_change
        elif action == 1:  # Sell
            reward = -price_change
        elif action == 2:  # Hold
            reward = 0.0
        
        return reward

if __name__ == "__main__":
    data_file = "data/processed_data/processed_data.csv"
    model_file = "models/trained_model.h5"
    
    trainer = ModelTrainer(data_file, model_file)
    trainer.preprocess_data()
    trainer.initialize_model()
    trainer.train_model()