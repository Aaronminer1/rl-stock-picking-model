# models/rl_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

class RLModel:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
    
    def build_model(self):
        """
        Build the reinforcement learning model.
        
        Returns:
            tf.keras.Model: The constructed RL model.
        """
        model = tf.keras.Sequential([
            LSTM(64, input_shape=(None, self.state_dim), return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(self.action_dim, activation='linear')
        ])
        
        return model
    
    def get_action(self, state):
        """
        Get the action for a given state using the RL model.
        
        Args:
            state (np.ndarray): The current state.
            
        Returns:
            np.ndarray: The predicted action.
        """
        state = np.reshape(state, (1, -1, self.state_dim))
        action = self.model.predict(state)
        return action[0]
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Perform a training step on a batch of experiences.
        
        Args:
            states (np.ndarray): Batch of states.
            actions (np.ndarray): Batch of actions.
            rewards (np.ndarray): Batch of rewards.
            next_states (np.ndarray): Batch of next states.
            dones (np.ndarray): Batch of done flags.
            
        Returns:
            float: The computed loss value.
        """
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            target_q_values = rewards + (1 - dones) * 0.99 * tf.reduce_max(self.model(next_states), axis=1)
            
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values_masked = tf.reduce_sum(q_values * action_masks, axis=1)
            
            loss = tf.keras.losses.mean_squared_error(target_q_values, q_values_masked)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save_model(self, file_path):
        """
        Save the trained model to a file.
        
        Args:
            file_path (str): Path to save the model.
        """
        self.model.save(file_path)
    
    def load_model(self, file_path):
        """
        Load a pre-trained model from a file.
        
        Args:
            file_path (str): Path to load the model from.
        """
        self.model = tf.keras.models.load_model(file_path)