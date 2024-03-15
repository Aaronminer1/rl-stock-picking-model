# strategies/technical_analysis.py

import numpy as np
import pandas as pd
from ta import add_all_ta_features
from base_strategy import BaseStrategy

class TechnicalAnalysisStrategy(BaseStrategy):
    def __init__(self, rl_model):
        super().__init__(rl_model)
    
    def get_action(self, state):
        """
        Get the action based on the current state using technical analysis.
        
        Args:
            state (np.ndarray): The current state.
            
        Returns:
            int: The selected action.
        """
        # Extract technical indicators from the state
        rsi = state[-1]
        sma_short = state[-2]
        sma_long = state[-3]
        
        # Implement your technical analysis logic here
        if rsi < 30 and sma_short > sma_long:
            action = 0  # Buy
        elif rsi > 70 and sma_short < sma_long:
            action = 1  # Sell
        else:
            action = 2  # Hold
        
        return action
    
    def train(self, data):
        """
        Train the strategy using the provided data.
        
        Args:
            data (pd.DataFrame): The training data.
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in range(len(data) - 1):
            state = data.iloc[i].values
            next_state = data.iloc[i + 1].values
            
            action = self.get_action(state)
            reward = self.calculate_reward(state, action, next_state)
            done = i == len(data) - 2
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        self.train_rl_model(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
    
    def evaluate(self, data):
        """
        Evaluate the strategy using the provided data.
        
        Args:
            data (pd.DataFrame): The evaluation data.
            
        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        states, actions, rewards = [], [], []
        
        for i in range(len(data) - 1):
            state = data.iloc[i].values
            next_state = data.iloc[i + 1].values
            
            action = self.get_action(state)
            reward = self.calculate_reward(state, action, next_state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        total_reward = sum(rewards)
        sharpe_ratio = self.calculate_sharpe_ratio(rewards)
        max_drawdown = self.calculate_max_drawdown(rewards)
        
        metrics = {
            'Total Reward': total_reward,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        
        return metrics
    
    def calculate_sharpe_ratio(self, rewards):
        """
        Calculate the Sharpe ratio based on the rewards.
        
        Args:
            rewards (list): The list of rewards.
            
        Returns:
            float: The calculated Sharpe ratio.
        """
        return np.mean(rewards) / np.std(rewards)
    
    def calculate_max_drawdown(self, rewards):
        """
        Calculate the maximum drawdown based on the rewards.
        
        Args:
            rewards (list): The list of rewards.
            
        Returns:
            float: The calculated maximum drawdown.
        """
        cumulative_returns = np.cumsum(rewards)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)