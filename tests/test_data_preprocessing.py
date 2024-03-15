# tests/test_data_preprocessing.py

import unittest
import pandas as pd
from data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.data = pd.DataFrame({
            'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
            'Open': [100, 105, 110],
            'High': [110, 115, 120],
            'Low': [90, 95, 100],
            'Close': [105, 110, 115],
            'Volume': [1000, 1500, 2000]
        })
        
    def test_preprocess_data(self):
        # Test the preprocess_data function
        preprocessed_data = preprocess_data(self.data)
        
        # Check if the preprocessed data has the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.assertEqual(list(preprocessed_data.columns), expected_columns)
        
        # Check if the preprocessed data has the expected shape
        expected_shape = (3, 5)
        self.assertEqual(preprocessed_data.shape, expected_shape)
        
        # Add more specific test cases as needed
        
if __name__ == '__main__':
    unittest.main()