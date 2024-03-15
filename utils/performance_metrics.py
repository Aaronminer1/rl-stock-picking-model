# utils/performance_metrics.py

import numpy as np
import pandas as pd

def calculate_returns(prices):
    returns = prices.pct_change()
    return returns

def calculate_cumulative_returns(returns):
    cumulative_returns = (1 + returns).cumprod() - 1
    return cumulative_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_max_drawdown(returns):
    cumulative_returns = calculate_cumulative_returns(returns)
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    return max_drawdown

def calculate_volatility(returns, annualization_factor=252):
    volatility = np.sqrt(annualization_factor) * returns.std()
    return volatility

def calculate_sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    sortino_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / downside_returns.std()
    return sortino_ratio