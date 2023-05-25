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

# Import the TA class from your code
from Ind import TA

# Create some sample data for testing
data = [10, 12, 15, 14, 13, 16, 18, 17, 19, 20]

# Test EMA function
ema = TA.EMA(data, 5)
# print("EMA:", ema)

# Test MACD function
# macd_line, signal_line, histogram = TA.MACD(data, 12, 26, 9)
# print("MACD Line:", macd_line)
# print("Signal Line:", signal_line)
# print("Histogram:", histogram)

# Test ATR function
# atr = TA.ATR(data, 5)
# print("ATR:", atr)

# # Test SMA function
# sma = TA.SMA(data, 5)
# print("SMA:", sma)

# # Test RSI function
# df = pd.DataFrame({"Close": data})
# rsi = TA.RSI(df, 5)
# print("RSI:", rsi)

# Test BB function
upper_band, middle_band, lower_band = TA.BB(data, 5, 2)
print("Upper Band:", upper_band)
print("Middle Band:", middle_band)
print("Lower Band:", lower_band)

# # Test stochastic_oscillator function
# stochastic_k, stochastic_d = TA.stochastic_oscillator(data, 5, 3)
# print("Stochastic %K:", stochastic_k)
# print("Stochastic %D:", stochastic_d)

# # Test ADX function
# data = [(12, 10, 11), (14, 11, 13), (16, 12, 15), (15, 12, 14), (14, 11, 13)]
# adx, plus_di, minus_di = TA.ADX(data, 4)
# print("ADX:", adx)
# print("Plus DI:", plus_di)
# print("Minus DI:", minus_di)

# # Test MA_envelope function
# upper_band, middle_band, lower_band = TA.MA_envelope(data, 5, 1)
# print("Upper Band:", upper_band)
# print("Middle Band:", middle_band)
# print("Lower Band:", lower_band)
