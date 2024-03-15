# models/model_evaluation.py

import numpy as np
import pandas as pd
from rl_model import RLModel
from data_preprocessing import preprocess_data
from config import CONFIG

class ModelEvaluator:
    def __init__(self, data_file, model_file):
        self.data_file = data_file
        self.model_file = model_file
        self.rl_model = None
        
        self.state_dim = CONFIG['state_dim']
        self.action_dim = CONFIG['action_dim']
        
    def preprocess_data(self):
        """
        Preprocess the evaluation data.
        """
        self.data = preprocess_data(self.data_file)
        
    def load_model(self):
        """
        Load the trained reinforcement learning model.
        """
        self.rl_model = RLModel(self.state_dim, self.action_dim)
        self.rl_model.load_model(self.model_file)
        
    def evaluate_model(self):
        """
        Evaluate the trained reinforcement learning model.
        """
        states = []
        actions = []
        rewards = []
        
        state = self.data.iloc[0].values
        done = False
        
        while not done:
            action = self.rl_model.get_action(state)
            next_state, reward, done = self.get_next_state_reward(state, action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards)
        
        self.calculate_metrics()
        
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
    
    def calculate_metrics(self):
        """
        Calculate evaluation metrics.
        """
        total_reward = np.sum(self.rewards)
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        
        print("Evaluation Metrics:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}")
    
    def calculate_sharpe_ratio(self):
        """
        Calculate the Sharpe ratio.
        
        Returns:
            float: The calculated Sharpe ratio.
        """
        returns = pd.Series(self.rewards).pct_change()
        sharpe_ratio = returns.mean() / returns.std()
        return sharpe_ratio
    
    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown.
        
        Returns:
            float: The calculated maximum drawdown.
        """
        cumulative_returns = np.cumsum(self.rewards)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        return max_drawdown

if __name__ == "__main__":
    data_file = "data/processed_data/test_data.csv"
    model_file = "models/trained_model.h5"
    
    evaluator = ModelEvaluator(data_file, model_file)
    evaluator.preprocess_data()
    evaluator.load_model()
    evaluator.evaluate_model()