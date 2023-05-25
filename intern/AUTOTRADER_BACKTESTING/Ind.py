import os
import sys
import time
import numpy as np
import pandas as pd
import numpy as np
import datetime as datetime
import traceback
# from tqdm import tqdm
from typing import Callable
from threading import Thread

"""
The TA (Technical Analysis) class contains a set of functions for performing technical analysis on market data. These functions include:

Exponential Moving Average (EMA): Calculates the exponential moving average of a given period of market data.

Moving Average Convergence Divergence (MACD): Calculates the MACD line, signal line, and histogram for a given period of market data, using a fast and slow period.

Average True Range (ATR): Calculates the average true range of a given period of market data.

Simple Moving Average (SMA): Calculates the simple moving average of a given period of market data.

Relative Strength Index (RSI): Calculates the relative strength index of a given period of market data.

Bollinger Bands (BB): Calculates the upper and lower Bollinger Bands of a given period of market data, based on a specified number of standard deviations.

Stochastic Oscillator: Calculates the stochastic oscillator of a given period of market data, using a specified period and simple moving average period.

Average Directional Index (ADX): Calculates the average directional index of a given period of market data.

Moving Average Envelope: Calculates the upper and lower moving average envelopes of a given period of market data, based on a specified number of standard deviations.

On-Balance Volume (OBV): Calculates the on-balance volume of a given period of market data.

Momentum (MOM): Calculates the momentum of a given period of market data.

Rate of Change (ROC): Calculates the rate of change of a given period of market data.

Beta (BETA): Calculates the beta coefficient of a given period of market data and a market index.

Correlation (CORREL): Calculates the correlation coefficient between two sets of market data for a given period.

Linear Regression (LINEARREG): Calculates the linear regression line of a given period of market data.

Linear Regression Angle (LINEARREG_ANGLE): Calculates the angle of the linear regression line of a given period of market data.

Linear Regression Intercept (LINEARREG_INTERCEPT): Calculates the y-intercept of the linear regression line of a given period of market data.

Linear Regression Slope (LINEARREG_SLOPE): Calculates the slope of the linear regression line of a given period of market data.

Standard Deviation (STDDEV): Calculates the standard deviation of a given period of market data.

"""

class TA: 
    """
    The EMA function calculates the Exponential Moving Average for a given data set and period.
    
    The function first calculates the alpha value using the period, which is the weight applied to 
    
    the current value of the data point. It then calculates the Simple Moving Average (SMA) for the first period data points and appends
    
    it to the ema list. Then, for the remaining data points, it calculates the EMA by taking the current data point 
    
    multiplied by the alpha value and adding it to the previous EMA multiplied by the complement of the alpha value. 
    
    This is done iteratively for all the remaining data points in the dataset. The function then returns the list of EMAs calculated.
    
    """
    def EMA(data, period):
        ema = []
        alpha = 2 / (period + 1)
        # print(data[:period],type(data[:period]))
        sma = sum(data[:period]) / period
        ema.append(sma)
        for i in range(period, len(data)):
            ema.append((data[i] * alpha) + (ema[-1] * (1 - alpha)))
        return ema

 

    """
    Calculate the fast EMA (Exponential Moving Average) by taking the average of the closing prices over the fast period.

    Calculate the slow EMA by taking the average of the closing prices over the slow period.

    Calculate the MACD line by subtracting the slow EMA from the fast EMA.

    Calculate the signal line by taking the average of the MACD line over the signal period.

    Calculate the histogram by subtracting the signal line from the MACD line.

    Return the MACD line, signal line, and histogram as outputs.

    """ 
    def MACD(data, fast_period, slow_period, signal_period):
        data_close = data['Close']
        ema_fast = TA.EMA(data_close, fast_period)
        ema_slow = TA.EMA(data_close, slow_period)
        # print(ema_fast,ema_slow)
        # print("len(ema_fast):-",len(ema_fast),"   len(ema_slow):-",len(ema_slow)," min:-",min(len(ema_slow),len(ema_fast)))
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(min(len(ema_slow),len(ema_fast)))]
        signal_line = TA.EMA(macd_line, signal_period)
        # print()
        histogram = [macd_line[i] - signal_line[i] for i in range(min(len(signal_line),len(macd_line)))]
        data['macd_line'] = 0
        data['signal_line'] = 0
        data['histogram'] = 0
        
        macd_line = [0] * (len(data['macd_line']) - len(macd_line)) + macd_line
        data['macd_line'] = macd_line
        signal_line = [0] * (len(data['signal_line']) - len(signal_line)) + signal_line
        data['signal_line'] = signal_line
        histogram = [0] * (len(data['histogram']) - len(histogram)) + histogram
        data['histogram'] = histogram
        # print("data:-",data) 
        return data
    
    """
    The ATR is calculated based on the true range (TR) of the price movement, 
    which is defined as the maximum of the following three values:
    
    The difference between the current high and the previous close
    
    The difference between the current low and the previous close
    
    The difference between the current high and the current low
    
    To calculate the ATR, we first calculate the TR for each period in the data. 
    
    We then calculate the average of the TRs over a certain period of time, 
    
    which is usually 14 days but can be adjusted depending on the trader's preferences
    """
    def ATR(data, period):
        tr = []
        for i in range(1, len(data)):
            high = data[i][0]
            low = data[i][1]
            prev_close = data[i-1][2]
            tr.append(max(high-low, abs(high-prev_close), abs(low-prev_close)))
        atr = TA.SMA(tr, period)
        return atr

    
    """
    The SMA (Simple Moving Average) function takes two parameters - data and period.
    
    data is the input time-series data that we want to calculate the moving average for and period is the number 
    of time periods we want to calculate the moving average over.
    
    First, we initialize an empty list sma_values to store the calculated moving average values. We then loop through the data list starting from the period index.

    Inside the loop, we use the sum() function to calculate the sum of the period number of values in the data list starting from the current index i
    
    and divide the sum by the period value to get the average. This gives us the SMA value for the current period.

    We then append the SMA value to the sma_values list and continue looping through the data list until we reach the end.

    Finally, we return the sma_values list containing all the calculated SMA values.   
    """    
    def SMA(data, period):
        sma = []
        for i in range(period-1, len(data)):
            sma.append(sum([data[j] for j in range(i-period+1, i+1)])/period)
        return sma
   
   
    """
    The RSI (Relative Strength Index) is calculated using the following steps in the RSI(data, period) function:
    
    Calculate the difference between the current price and the previous price: delta = data.diff().dropna()
    
    Separate the positive differences from the negative differences: gain = delta.where(delta > 0, 0),
    loss = -delta.where(delta < 0, 0)
    
    Calculate the average gain and average loss over the specified 
    
    period: avg_gain = gain.rolling(window=period).mean(), avg_loss = loss.rolling(window=period).mean()
    
    Calculate the Relative Strength: rs = avg_gain / avg_loss
    
    Calculate the RSI: rsi = 100.0 - (100.0 / (1.0 + rs))
    """
    def RSI(data, period):
        
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    """
    The BB (Bollinger Bands) function with parameters BB(data, period, n_std) is calculated using the following steps:

    Calculate the rolling mean over the specified period: sma = data.rolling(window=period).mean()

    Calculate the rolling standard deviation over the specified period: std = data.rolling(window=period).std()

    Calculate the upper band: upper_band = sma + (n_std * std)

    Calculate the lower band: lower_band = sma - (n_std * std)

    Calculate the middle band: middle_band = sma

    Return the upper, lower, and middle bands as a tuple: (upper_band, middle_band, lower_band)
    """
    def BB(data, period, n_std):
        data_close = data['Close']
        sma =TA.SMA(data_close, period)
        std = np.std(data_close[-period:])
        upper_band = [sma[i] + n_std*std for i in range(len(sma))]
        lower_band = [sma[i] - n_std*std for i in range(len(sma))]
        data['upper_band'] = 0
        data['sma'] = 0
        data['lower_band'] = 0
        upper_band = [0] * (len(data['upper_band']) - len(upper_band)) + upper_band
        sma = [0] * (len(data['sma']) - len(sma)) + sma        
        lower_band = [0] * (len(data['lower_band']) - len(lower_band)) + lower_band
        data['upper_band'] = upper_band
        data['sma'] = sma
        data['lower_band'] = lower_band
        return data
    """
    The Stochastic Oscillator is calculated using the following steps in the stochastic_oscillator(data, period, sma_period) function:
    
    Calculate the highest high and lowest low over the specified period: 
    
    highest_high = data['High'].rolling(window=period).max(), lowest_low = data['Low'].rolling(window=period).min()
    
    Calculate the %K: (data['Close'] - lowest_low) / (highest_high - lowest_low) * 100
    
    Calculate the %D using a simple moving average over the specified period: %D = %K.rolling(window=sma_period).mean()
    
    The Stochastic Oscillator value oscillates between 0 and 100, and is usually plotted on a chart with horizontal lines drawn at the 20 and 80 levels. A reading above 80 is considered overbought and a reading below 20 is considered oversold.
    """
    def stochastic_oscillator(data, period, sma_period):
        high = [d[0] for d in data]
        low = [d[1] for d in data]
        close = [d[2] for d in data]
        
        K = []
        for i in range(period-1, len(data)):
            c = close[i]
            l = min(low[i-period+1:i+1])
            h = max(high[i-period+1:i+1])
            k = (c-l)/(h-l)*100
            K.append(k)
        
        D = TA.SMA(K, sma_period)
        
        return K, D

    """
    ADX function calculates the Average Directional Index (ADX) along with the Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI).
    
    It takes in a data list which contains the high, low, and close prices for a certain period, as well as a period value for the ATR and other calculations.

    First, the function extracts the high, low, and close prices from the data list. Then, it calculates the True Range (TR) for each period using the maximum of 
    
    the difference between the high and low, the absolute value of the difference between the high and the previous close, and the absolute value of the difference between the low and the previous close.
    
    The TR is then smoothed using the Simple Moving Average (SMA) with the period provided in the function arguments to obtain the Average True Range (ATR).

    Next, the function calculates the Plus Directional Movement (+DM) and Minus Directional Movement (-DM) using the differences between consecutive highs and lows, depending on which one is greater,
    
    and sets any negative values to zero. The +DM and -DM are then smoothed using the SMA with the period provided in the function arguments to obtain the Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI).

    Using the +DI and -DI, the function calculates the Directional Movement Index (DX) which represents the strength of the trend. 
    
    Finally, the DX is smoothed using the SMA with the period provided in the function arguments to obtain the ADX value. 
    
    The function returns the ADX, +DI, and -DI values as a tuple.
    """
    def ADX(data, period):
        high = [d[0] for d in data]
        low = [d[1] for d in data]
        close = [d[2] for d in data]
        
        tr = []
        for i in range(1, len(data)):
            h = high[i]
            l = low[i]
            pc = close[i-1]
            tr.append(max(h-l, abs(h-pc), abs(l-pc)))
        ATR = TA.SMA(tr, period)
        
        up_move = [high[i] - high[i-1] for i in range(1, len(high)) if high[i] > high[i-1]]
        down_move = [low[i-1] - low[i] for i in range(1, len(low)) if low[i] < low[i-1]]
        plus_DM = [0] * (len(down_move)-len(up_move)) + [max(0, up_move[i]-down_move[i]) for i in range(len(up_move))]
        minus_DM = [0] * (len(up_move)-len(down_move)) + [max(0, down_move[i]-up_move[i]) for i in range(len(down_move))]
        plus_DI = TA.SMA(plus_DM, period) / ATR
        minus_DI = TA.SMA(minus_DM, period) / ATR
        DX = abs(plus_DI - minus_DI) / (plus_DI + minus_DI) * 100
        ADX = TA.SMA(DX, period)
        return ADX, plus_DI, minus_DI


    """
    The MA_envelope function calculates the Moving Average Envelope for a given data list, using a specified period and number of standard deviations. 

    The envelope consists of an upper and lower band that are a certain percentage above and below the moving average.

    The function first calculates the Simple Moving Average (SMA) of the data using the period provided in the function arguments. 

    It then calculates the upper and lower bands by multiplying the SMA by a certain number of standard deviations (n_std) and adding or subtracting the result from the SMA. This creates two lines that represent a certain percentage (n_std) above and below the SMA.
    """
    def MA_envelope(data, period, n_std):
        sma = TA.SMA(data, period)
        std = np.std(data[-period:])
        
        upper_band = [sma[i] + n_std*std for i in range(len(sma))]
        lower_band = [sma[i] - n_std*std for i in range(len(sma))]
        
        return upper_band, sma, lower_band



    # On Balance Volume (OBV):
    def OBV(data):
        """
        The OBV function calculates the On Balance Volume indicator for a given dataset.
        
        It iterates over the data points and calculates the OBV value based on the volume of the 
        current data point and the direction of the price movement relative to the previous data point.
        If the current data point's closing price is higher than the previous data point's closing price,
        the OBV is increased by the volume of the current data point. If the current data point's closing 
        price is lower than the previous data point's closing price, the OBV is decreased by the volume 
        of the current data point. If the current data point's closing price is equal to the previous 
        data point's closing price, the OBV remains the same. The function returns the list of OBV values.
        
        Parameters:
        data (list): List of OHLCV (Open-High-Low-Close-Volume) data points.
        
        Returns:
        list: The list of OBV values calculated.
        """
        obv = [0]
        for i in range(1, len(data)):
            if data[i][2] > data[i-1][2]:
                obv.append(obv[-1] + data[i][3])
            elif data[i][2] < data[i-1][2]:
                obv.append(obv[-1] - data[i][3])
            else:
                obv.append(obv[-1])
        return obv

    """
    MOM - Momentum Indicator Calculation

    Calculates the momentum of the close prices over a specified period of time.
     
    Parameters:
         data (list of tuples): A list of tuples where each tuple represents a single period of OHLC (Open, High, Low, Close) data.
         period (int): The number of periods over which to calculate momentum.
    
    Returns:
         list: A list of momentum values calculated using the formula:
               mom = close_price[i] - close_price[i-period]
    
    """ 
    def MOM(data, period):
        close = [d[2] for d in data]
        mom = [close[i] - close[i-period] for i in range(period, len(close))]
        return mom
 
 
    """
    The ROC function calculates the Rate of Change for a given data set and period.

    The function first creates a list of close prices from the given dataset.

    Then, for each data point after the period, it calculates the ROC by taking the difference between the current
    close price and the close price from "period" days ago, then dividing that difference by the close price from "period"
    days ago, and finally multiplying by 100 to convert to a percentage change.

    This is done iteratively for all the remaining data points in the dataset. The function then returns the list of ROCs calculated.

    """
    
    def ROC(data, period):
        """
        Calculates the rate of change for a given period using the close price.

        Args:
        - data: a list of OHLCV data in the form [[timestamp, open, high, low, close, volume], ...]
        - period: an integer representing the number of periods to calculate ROC for

        Returns:
        - roc: a list of the rate of change values for the given period
        """
        close = [d[4] for d in data] # select close prices from OHLCV data
        roc = [(close[i] - close[i-period])/close[i-period] * 100 for i in range(period, len(close))]
        return roc


    # Beta (BETA):
    def BETA(data, market_data, period):
        """
        Calculates the beta value for a given period using the close prices of two data sets.

        Args:
        - data: a list of OHLCV data in the form [[timestamp, open, high, low, close, volume], ...]
        - market_data: a list of OHLCV data for the market in the same format
        - period: an integer representing the number of periods to calculate beta for

        Returns:
        - beta: a list of the beta values for the given period
        """
        close = [d[4] for d in data] # select close prices from OHLCV data
        market_close = [d[4] for d in market_data] # select close prices from market OHLCV data
        beta = []
        for i in range(period, len(close)):
            covar = np.cov(close[i-period:i+1], market_close[i-period:i+1])[0][1]
            market_var = np.var(market_close[i-period:i+1])
            beta.append(covar/market_var)
        return beta


    # Pearson's Correlation Coefficient (CORREL):
    def CORREL(data1, data2, period):
        """
        Calculates the Pearson's correlation coefficient for a given period using the close prices of two data sets.

        Args:
        - data1: a list of OHLCV data in the form [[timestamp, open, high, low, close, volume], ...]
        - data2: a list of OHLCV data for another data set in the same format
        - period: an integer representing the number of periods to calculate correlation for

        Returns:
        - corr: a list of the correlation coefficient values for the given period
        """
        close1 = [d[4] for d in data1] # select close prices from first OHLCV data
        close2 = [d[4] for d in data2] # select close prices from second OHLCV data
        corr = []
        for i in range(period, len(close1)):
            corr.append(np.corrcoef(close1[i-period:i+1], close2[i-period:i+1])[0][1])
        return corr


    #Linear Regression (LINEARREG):
    """
    # Linear Regression (LR) function
    # Calculates the linear regression over a given period for a list of closing prices
    # Inputs:
    #   data: list of OHLCV data for a given security or asset
    #   period: integer indicating the number of periods to use for the linear regression calculation
    # Returns:
    #   list of LR values over the given period
    """
    def LINEARREG(data, period):
        close = [d[2] for d in data]
        lr = []
        for i in range(period, len(close)):
            x = np.arange(0, period)
            y = close[i-period:i]
            slope, intercept = np.polyfit(x, y, 1)
            lr.append((slope*period) + intercept)
        return lr
  


    """
    # Linear Regression Angle (LINEARREG_ANGLE) function
    # Calculates the angle of the linear regression over a given period for a list of closing prices
    # Inputs:
    #   data: list of OHLCV data for a given security or asset
    #   period: integer indicating the number of periods to use for the linear regression angle calculation
    # Returns:
    #   list of LR angle values over the given period

    """
    #Linear Regression Angle (LINEARREG_ANGLE):
    def LINEARREG_ANGLE(data, period):
        close = [d[2] for d in data]
        angle = []
        for i in range(period, len(close)):
            x = np.arange(0, period)
            y = close[i-period:i]
            slope, _ = np.polyfit(x, y, 1)
            angle.append(np.degrees(np.arctan(slope)))
        return angle



    #Linear Regression Intercept (LINEARREG_INTERCEPT):

    def LINEARREG_INTERCEPT(data, period):
        close = [d[2] for d in data]
        intercept = []
        for i in range(period, len(close)):
            x = np.arange(0, period)
            y = close[i-period:i]
            slope, intercept_ = np.polyfit(x, y, 1)
            intercept.append(intercept_)
        return intercept



    #Linear Regression Slope (LINEARREG_SLOPE):

    def LINEARREG_SLOPE(data, period):
        close = [d[2] for d in data]
        slope = []
        for i in range(period, len(close)):
            x = np.arange(0, period)
            y = close[i-period:i]
            slope_, _ = np.polyfit(x, y, 1)
            slope.append(slope_)
        return slope
        
        
    #Standard Deviation (STDDEV):
    def STDDEV(data, period):
        close = [d[2] for d in data]
        stddev = []
        for i in range(period, len(close)):
            stddev.append(np.std(close[i-period:i]))
        return stddev