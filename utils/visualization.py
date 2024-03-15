# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_stock_prices(prices):
    plt.figure(figsize=(12, 6))
    plt.plot(prices)
    plt.title('Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def plot_stock_returns(returns):
    plt.figure(figsize=(12, 6))
    plt.plot(returns)
    plt.title('Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.show()

def plot_cumulative_returns(cumulative_returns):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()

def plot_Portfolio_weights(weights):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weights.index, y=weights.values)
    plt.title('Portfolio Weights')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.show()

def plot_portfolio_performance(portfolio_returns, benchmark_returns):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_returns, label='Portfolio')
    plt.plot(benchmark_returns, label='Benchmark')
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()