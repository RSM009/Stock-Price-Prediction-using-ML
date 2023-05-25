import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
# Define the ticker symbol
ticker_symbol = 'AUDUSD=X'

# Get the data from Yahoo Finance
stock_data = yf.download(tickers=ticker_symbol, interval='1m', start="2023-05-17",end="2023-05-18")

trace = [go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'])]
# Create the candlestick chart
layout = go.Layout(title=ticker_symbol + ' Live Candlestick Chart',
                   xaxis_title='Date',
                   yaxis_title='Price')

fig = go.Figure(data=trace,layout=layout)
fig.show()
var =0
while True:
    new_data = yf.download(tickers=ticker_symbol, interval='1m', start="2023-05-17",end="2023-05-18")
    new_trace = go.Candlestick(x=new_data.index,
                            open=new_data['Open'],
                            high=new_data['High'],
                            low=new_data['Low'],
                            close=new_data['Close'])

    fig.update(data=[new_trace])

    # var +=1
    # Display th chart
    
