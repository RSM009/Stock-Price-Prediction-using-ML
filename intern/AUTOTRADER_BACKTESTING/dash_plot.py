import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import yfinance as yf
import feed as ly
from Ind import TA
import talib as ta 
X = deque(maxlen = 10)
X.append(1)

Y = deque(maxlen = 20)
Y.append(1)

app = dash.Dash(__name__)

app.layout = html.Div(
	[
		dcc.Graph(id = 'live-graph', animate = True),
		dcc.Interval(
			id = 'graph-update',
			interval = 100,
			n_intervals = 0
		),
	],
 style={'width' :'100%', 'height' : '500px'}
)

@app.callback(
	Output('live-graph', 'figure'),
	[ Input('graph-update', 'n_intervals') ]
)

def update_graph_scatter(n):
    # Call get_live_data() to get the latest tick data for the stock symbol
    tick_data = ly.get_live_data('AUDUSD=X')
    # tick_data = ly.converter_tick_to_candle_format(tick_data=tick_data,time_frame=1)
    indicator_data = TA.RSI(tick_data,14)
    # print(indicator_data)
    
    # Create the candlestick chart data
    data = [go.Candlestick(x=tick_data.index,
                           open=tick_data['Open'],
                           high=tick_data['High'],
                           low=tick_data['Low'],
                           close=tick_data['Close'])]
    
    layout = go.Layout(
        xaxis=dict(range=[min(tick_data.index), max(tick_data.index)],),
        yaxis=dict(range=[min(tick_data['Low']), max(tick_data['High'])]),
#       yaxis=dict(range=[18300, 18400]),
      
    )
    
    return {'data': data, 'layout': layout}
if __name__ == '__main__':
	app.run_server()
