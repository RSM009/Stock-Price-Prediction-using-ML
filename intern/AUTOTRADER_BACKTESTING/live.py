import time
import psutil
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import yfinance as yf


def update_candle_chart(data, ax):
    data = data.tail(50)  # keep only the last 50 rows for display
    # Create a new candlestick plot
    mpf.plot(data, type='candle', style='charles', volume=False, ax=ax)
    # Update the plot
    ax.set_xlim(left=data.index.min(), right=data.index.max())
    ax.set_xticks(data.index)
    ax.set_xticklabels([d.strftime('%m-%d %H:%M') for d in data.index], rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True)
    # Redraw thecanvas
    fig.canvas.draw()
    plt.show()


symbol = "AAPL"
data = pd.DataFrame()
fig = plt.figure()
ax = fig.add_subplot(111)
var = 0
while True:
    new_data = yf.download(symbol, interval='1d', start="2023-01-01")
    if new_data.empty:
        continue  # no new data available
    data = pd.concat([data, new_data])
    update_candle_chart(data, ax)
    time.sleep(60)  # wait for 1 minute before fetching new data