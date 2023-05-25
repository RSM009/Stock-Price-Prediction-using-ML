import yfinance as yf
import mplfinance as mpf
import pandas as pd
import threading
from Ind import TA
import time

def get_live_data(symbol,start_date,end_date,inter):
    tick_data = yf.download(symbol, interval=inter, start=str(start_date),end=end_date)
    return tick_data

def get_live_tick_data(symbol,start_date,end_date,inter):
    """Queries yfinance every second to get the latest tick data."""
    tick_data = yf.download(symbol, interval=inter, start=str(start_date),end=str(end_date))
    tick = tick_data['Close']
    tick_data = tick.tail(1)
    return tick_data

# print(get_live_tick_data("BTC-USD"))

def converter_tick_to_candle_format(tick_data, time_frame):
    """
    Converts tick data for a stock to the desired time frame (in minutes).
    Returns a DataFrame with columns for open, low, high, and close prices.
    """
    # Resample the tick data to the desired time frame
    resampled = tick_data.resample(f"{time_frame}T").agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    
    # Drop any rows with missing data
    resampled.dropna(inplace=True)
    
    return resampled
   

# while 1:
#     data = get_live_data("BTC-USD")
#     x = TA.RSI(data=data,period=14)
#     print("x:-",x)


