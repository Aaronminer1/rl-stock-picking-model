# Stock Picking Reinforcement Learning Model
This project was completly coded using GPT-4 in colaboration with Claude 3 Opus. GPT-4 helped with the outline and critiquing of the project and claude3 Opus supplyed the code and file structure. This is an ongoing experimental project between the llm's to show the potentail of leveraging the strengths of different models for project building. The code has not been tested that is the next phase with recurent update and upgrades as we go along. 


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

### Local Installation

1. Clone the repository:
git clone git@github.com:Aaronminer1/rl-stock-picking-model.git


Copy code

2. Navigate to the project directory:
cd rl-stock-picking-model


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


Copy code

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch to your forked repository.
4. Submit a pull request to the main repository.

Please ensure that your code follows the project's coding style and conventions, and includes appropriate tests and documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or inquiries, please contact the project maintainer:

Aaron Miner
- Email: your-email@example.com
- GitHub: [Aaronminer1](https://github.com/Aaronminer1)