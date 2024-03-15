# config.py

CONFIG = {
    # Data preprocessing parameters
    'data': {
        'raw_data_path': 'data/raw_data',
        'processed_data_path': 'data/processed_data',
        'train_data_file': 'train_data.csv',
        'test_data_file': 'test_data.csv',
        'columns_to_scale': ['Close', 'Volume'],
        'columns_to_encode': ['Sector']
    },
    
    # Reinforcement learning model parameters
    'model': {
        'state_dim': 30,
        'action_dim': 3,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    },
    
    # Training parameters
    'training': {
        'num_episodes': 100,
        'max_steps_per_episode': 1000,
        'checkpoint_path': 'checkpoints/rl_model.pt'
    },
    
    # Evaluation parameters
    'evaluation': {
        'num_episodes': 10,
        'max_steps_per_episode': 1000
    },
    
    # Trading strategy parameters
    'strategy': {
        'initial_balance': 10000,
        'commission_rate': 0.001,
        'technical_indicators': ['RSI', 'SMA', 'EMA'],
        'fundamental_factors': ['P/E', 'EPS', 'Debt/Equity']
    }
}