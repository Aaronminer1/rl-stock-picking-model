# strategies/base_strategy.py

from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, rl_model):
        self.rl_model = rl_model
    
    @abstractmethod
    def get_action(self, state):
        """
        Get the action based on the current state using the specific strategy.
        
        Args:
            state (np.ndarray): The current state.
            
        Returns:
            int: The selected action.
        """
        pass
    
    @abstractmethod
    def train(self, data):
        """
        Train the strategy using the provided data.
        
        Args:
            data (pd.DataFrame): The training data.
        """
        pass
    
    @abstractmethod
    def evaluate(self, data):
        """
        Evaluate the strategy using the provided data.
        
        Args:
            data (pd.DataFrame): The evaluation data.
            
        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        pass
    
    def get_rl_action(self, state):
        """
        Get the action from the reinforcement learning model.
        
        Args:
            state (np.ndarray): The current state.
            
        Returns:
            int: The selected action.
        """
        action = self.rl_model.get_action(state)
        return np.argmax(action)
    
    def train_rl_model(self, states, actions, rewards, next_states, dones):
        """
        Train the reinforcement learning model using the provided data.
        
        Args:
            states (np.ndarray): The array of states.
            actions (np.ndarray): The array of actions.
            rewards (np.ndarray): The array of rewards.
            next_states (np.ndarray): The array of next states.
            dones (np.ndarray): The array of done flags.
        """
        self.rl_model.train_step(states, actions, rewards, next_states, dones)
    
    def calculate_reward(self, state, action, next_state):
        """
        Calculate the reward based on the current state, action, and next state.
        
        Args:
            state (np.ndarray): The current state.
            action (int): The selected action.
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