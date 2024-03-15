# utils/alpaca_api.py

import os
import alpaca_trade_api as tradeapi

def get_alpaca_api():
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    base_url = os.getenv('ALPACA_BASE_URL')

    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    return api

def get_account_info():
    api = get_alpaca_api()
    account_info = api.get_account()
    return account_info

def get_position(symbol):
    api = get_alpaca_api()
    position = api.get_position(symbol)
    return position

def get_market_data(symbol, timeframe, limit):
    api = get_alpaca_api()
    market_data = api.get_barset(symbol, timeframe, limit=limit).df
    return market_data

def submit_order(symbol, qty, side, type, time_in_force):
    api = get_alpaca_api()
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=type,
        time_in_force=time_in_force
    )
    return order