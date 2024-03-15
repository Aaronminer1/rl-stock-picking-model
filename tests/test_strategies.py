# tests/test_strategies.py

import unittest
import pandas as pd
from strategies.technical_analysis import TechnicalAnalysisStrategy
from strategies.fundamental_analysis import FundamentalAnalysisStrategy
from rl_model import RLModel

class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Create instances of the strategies and RL model for testing
        self.rl_model = RLModel(state_dim=5, action_dim=3)
        self.technical_strategy = TechnicalAnalysisStrategy(self.rl_model)
        self.fundamental_strategy = FundamentalAnalysisStrategy(self.rl_model)
        
        # Create sample data for testing
        self.data = pd.DataFrame({
            'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'Open': [100, 105, 110],
            'High': [110, 115, 120],
            'Low': [90, 95, 100],
            'Close': [105, 110, 115],
            'Volume': [1000, 1500, 2000],
            'RSI': [50, 60, 70],
            'SMA': [100, 105, 110],
            'EMA': [100, 103, 107],
            'P/E': [15, 16, 17],
            'EPS': [2, 2.1, 2.2]
        })
        
    def test_technical_strategy(self):
        # Test the technical analysis strategy
        self.technical_strategy.train(self.data)
        metrics = self.technical_strategy.evaluate(self.data)
        
        # Check if the returned metrics are a dictionary
        self.assertIsInstance(metrics, dict)
        
        # Add more specific test cases as needed
        
    def test_fundamental_strategy(self):
        # Test the fundamental analysis strategy
        self.fundamental_strategy.train(self.data)
        metrics = self.fundamental_strategy.evaluate(self.data)
        
        # Check if the returned metrics are a dictionary
        self.assertIsInstance(metrics, dict)
        
        # Add more specific test cases as needed
        
if __name__ == '__main__':
    unittest.main()