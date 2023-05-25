import yfinance as yf
import mplfinance as mpf
import pandas as pd
import threading
import time

def get_live_data(symbol):
    """Queries yfinance every second to get the latest tick data."""
    tick_data = yf.download(symbol, interval='1m', start="2023-05-12",end="2023-05-14")
    tick = tick_data['Close']
    tick_data = tick_data.iloc[-1]
    # print(tick_data)
    return tick_data

# get_live_data("AUDUSD=X")
# def plot_live_candlestick(data):
#     """Plots a live candlestick chart using mplfinance."""
#     # Convert tick data to OHLC data
#     print("data:-",data)
#     ohlc_data = []
#     ohlc_data = data.resample('1min').agg({
#         'Open': 'first',
#         'High': 'max',
#         'Low': 'min',
#         'Close': 'last',
#         'Volume': 'sum'
#     }).dropna()
#     print("ohlc_data:-",ohlc_data)
    
#     # Plot live candlestick chart
#     mpf.plot(ohlc_data, type='candle', style='charles', volume=True,
#              title='Live Candlestick Chart', ylabel='Price', ylabel_lower='Volume')
def tick_to_ohlc(tick_data):
    """Converts tick data to OHLC data."""
    ohlc_data = tick_data.resample('1min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return ohlc_data 
    

# Start a new thread to get live data every second
symbol = 'AUDUSD=X'

dx = get_live_data(symbol=symbol)
# print(dx['Open'])
print(dx)
data = {'Open': dx['Open'],
        'High': dx['High'],
        'Low': dx['Low'],
        'Close': dx['Close'],
        'Adj Close': ['Adj Close'],
        }
print(data)
df = pd.DataFrame(data)

print(df)
mpf.plot(df, type='candle', style='charles', volume=True,title='Live Candlestick Chart', ylabel='Price', ylabel_lower='Volume')


# df = pd.DataFrame([dx])
# print("dx:-",dx,type(df))
# print(tick_to_ohlc(tick_data=df))
# data_thread = threading.Thread(target=get_live_data, args=(symbol,))
# data_thread.start()

# # Plot live candlestick chart in main thread
# tick_data = pd.DataFrame()  # initialize an empty DataFrame
# plot_live_candlestick(tick_data)
