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
from plotly.subplots import make_subplots
# import talib as ta 
X = deque(maxlen = 10)
X.append(1)

Y = deque(maxlen = 20)
Y.append(1)

app = dash.Dash(__name__)

app.layout = html.Div(
	[
		html.Div(
            dcc.Graph(id='candlestick-graph', animate=True),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Graph(id='indicator-graph_rsi', animate=True),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Graph(id='indicator-graph_bolinger', animate=True),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        dcc.Interval(
            id='graph-update',
            interval=100,
            n_intervals=0
        ),
	],
 style={'width' :'100%', 'height' : '500px'}
)
@app.callback(
    Output('candlestick-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_candlestick_graph(n):
    # symbol = input("write the symbol for data:-")
    # start_date = input("write the start_datw for data(YYYY-MM-DD):-")
    # end_date = input("write the end_date for data(YYYY-MM-DD):-")
    # inter = input("write the Interval for data(1m,5m,15m,1d,1w):-")
    tick_data = ly.get_live_data(symbol="^NSEBANK",start_date="2023-05-19",end_date="2023-05-20",inter="1m")
    
    # Create the candlestick chart data
    candlestick_data = go.Candlestick(x=tick_data.index,
                                      open=tick_data['Open'],
                                      high=tick_data['High'],
                                      low=tick_data['Low'],
                                      close=tick_data['Close'])
    
    layout = go.Layout(
        xaxis=dict(range=[min(tick_data.index), max(tick_data.index)],),
        yaxis=dict(range=[min(tick_data['Low']), max(tick_data['High'])]),
    )
    
    return {'data': [candlestick_data], 'layout': layout}



@app.callback(
    Output('indicator-graph_rsi', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_indicator_graph_rsi(n):
    # symbol = input("write the symbol for data:-")
    # start_date = input("write the start_datw for data(YYYY-MM-DD):-")
    # end_date = input("write the end_date for data(YYYY-MM-DD):-")
    # inter = input("write the Interval for data(1m,5m,15m,1d,1w):-")
    tick_data = ly.get_live_data(symbol="^NSEBANK",start_date="2023-05-19",end_date="2023-05-20",inter="1m")
    rsi_data = TA.RSI(data=tick_data,period=14)
    rsi_trace = go.Scatter(x=tick_data.index, y=rsi_data, name='RSI')
    
    layout = go.Layout(
        xaxis=dict(range=[min(tick_data.index), max(tick_data.index)],),
        yaxis=dict(range=[0,100]),
    #    yaxis=dict(range=[18300, 18400]),
    )
    return {'data': [rsi_trace], 'layout': layout}







@app.callback(
    Output('indicator-graph_bolinger', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_indicator_graph_bb(n):
    
    tick_data = ly.get_live_data(symbol="^NSEBANK",start_date="2023-05-18",end_date="2023-05-20",inter="1m")
    # data = 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Candlestick(
        x=tick_data.index,
        open=tick_data['Open'],
        high=tick_data['High'],
        low=tick_data['Low'],
        close=tick_data['Close']
    ))

    data = TA.BB(data = tick_data,period=10,n_std=1)
    upper_band = data["upper_band"]
    middle_band = data["sma"]
    lower_band  = data["lower_band"]
    fig.add_trace(go.Scatter(
        x=tick_data.index,
        y=upper_band,
        mode='lines',
        name='Upper Band'
    ))

    fig.add_trace(go.Scatter(
        x=tick_data.index,
        y=middle_band,
        mode='lines',
        name='Middle Band'
    ))

    fig.add_trace(go.Scatter(
        x=tick_data.index,
        y=lower_band,
        mode='lines',
        name='Lower Band'
    ))

    # layout = go.Layout(
    #     xaxis=dict(range=[min(tick_data.index), max(tick_data.index)]),
    #     yaxis=dict(range=[min(tick_data['Low']), max(tick_data['High'])]),
    # )
    layout = go.Layout(
        xaxis=dict(range=[min(tick_data.index), max(tick_data.index)],),
        yaxis=dict(range=[min(tick_data['Low']), max(tick_data['High'])]),
    #    yaxis=dict(range=[18300, 18400]),
    )
    return {'data': fig.data, 'layout': layout}
  



      
if __name__ == '__main__':
	app.run_server()
