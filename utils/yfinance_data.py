# utils/yfinance_data.py

import yfinance as yf

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def get_stock_info(symbol):
    stock = yf.Ticker(symbol)
    stock_info = stock.info
    return stock_info

def get_stock_financials(symbol):
    stock = yf.Ticker(symbol)
    financials = stock.financials
    return financials

def get_stock_actions(symbol):
    stock = yf.Ticker(symbol)
    actions = stock.actions
    return actions