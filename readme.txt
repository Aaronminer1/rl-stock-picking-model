# Stock Picking Reinforcement Learning Model

This project implements a reinforcement learning model for stock picking and trading. The model learns to make trading decisions based on historical stock data and real-time market information. It incorporates technical analysis and fundamental analysis strategies to generate trading signals.

## Features

- Reinforcement learning model for stock trading
- Integration with Alpaca API for real-time market data and trading execution
- Integration with yfinance for historical stock data retrieval
- Technical analysis and fundamental analysis trading strategies
- Comprehensive evaluation metrics and performance tracking
- Visualization utilities for analyzing model behavior and performance
- Modular and extensible codebase with unit testing

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Windows

1. Clone the repository:
git clone https://github.com/your-username/stock-picking-rl-model.git


Copy code

2. Navigate to the project directory:
cd stock-picking-rl-model


Copy code

3. Create a virtual environment:
python -m venv venv


Copy code

4. Activate the virtual environment:
venv\Scripts\activate


Copy code

5. Install the required dependencies:
pip install -r requirements.txt


Copy code

### Linux and macOS

1. Clone the repository:
git clone https://github.com/your-username/stock-picking-rl-model.git


Copy code

2. Navigate to the project directory:
cd stock-picking-rl-model


Copy code

3. Create a virtual environment:
python3 -m venv venv


Copy code

4. Activate the virtual environment:
source venv/bin/activate


Copy code

5. Install the required dependencies:
pip install -r requirements.txt


Copy code

## Configuration

1. Rename the `config.example.py` file to `config.py`.

2. Open the `config.py` file and update the following configuration settings:
- Alpaca API credentials: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_API_URL`
- Data paths: `DATA_DIR`, `MODEL_DIR`
- Model hyperparameters: `STATE_DIM`, `ACTION_DIM`, `LEARNING_RATE`, etc.

## Usage

1. Preprocess the historical stock data:
python preprocess_data.py


Copy code

2. Train the reinforcement learning model:
python train_model.py


Copy code

3. Evaluate the trained model:
python evaluate_model.py


Copy code

4. Run the trading strategy:
python run_strategy.py


Copy code

## Testing

To run the unit tests, execute the following command:
python -m unittest discover tests