# main.py

import os
import pandas as pd
from config import CONFIG
from data_preprocessing import preprocess_data
from rl_model import RLModel
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from strategies.technical_analysis import TechnicalAnalysisStrategy
from strategies.fundamental_analysis import FundamentalAnalysisStrategy

def load_data():
    train_data_path = os.path.join(CONFIG['data']['processed_data_path'], CONFIG['data']['train_data_file'])
    test_data_path = os.path.join(CONFIG['data']['processed_data_path'], CONFIG['data']['test_data_file'])
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    return train_data, test_data

def create_model():
    state_dim = CONFIG['model']['state_dim']
    action_dim = CONFIG['model']['action_dim']
    learning_rate = CONFIG['model']['learning_rate']
    
    model = RLModel(state_dim, action_dim, learning_rate)
    return model

def train_model(model, train_data):
    trainer = ModelTrainer(model, CONFIG['training'])
    trainer.train(train_data)

def evaluate_model(model, test_data):
    evaluator = ModelEvaluator(model, CONFIG['evaluation'])
    metrics = evaluator.evaluate(test_data)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def execute_strategy(model, strategy_name, data):
    if strategy_name == 'technical':
        strategy = TechnicalAnalysisStrategy(model)
    elif strategy_name == 'fundamental':
        strategy = FundamentalAnalysisStrategy(model)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy.train(data)
    metrics = strategy.evaluate(data)
    print(f"\n{strategy_name.capitalize()} Analysis Strategy Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def main():
    # Load and preprocess data
    train_data, test_data = load_data()
    train_data = preprocess_data(train_data, CONFIG['data'])
    test_data = preprocess_data(test_data, CONFIG['data'])
    
    # Create and train the model
    model = create_model()
    train_model(model, train_data)
    
    # Evaluate the model
    evaluate_model(model, test_data)
    
    # Execute trading strategies
    execute_strategy(model, 'technical', test_data)
    execute_strategy(model, 'fundamental', test_data)

if __name__ == '__main__':
    main()