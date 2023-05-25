import feed as fd
from Ind import TA
import pandas as pd 
# data = fd.get_live_data("^NSEBANK")
# print(data)
import csv

def append_series_to_csv_rsi( series):
    filename = "result.txt"
    file_exists = pd.io.common.file_exists(filename)

    # Extract required values from the series
    values = [
        series.name,  # Datetime
        series['Close'],
        series['RSi'],
        series['Position']
    ]

    # Create a new DataFrame with column names
    
    df = pd.DataFrame([values], columns=['Datetime','Position', 'Close', 'RSi'])

    # Write the DataFrame to the CSV file
    df.to_csv(filename, mode='a', header=True,sep=" ", index=False)
def append_series_to_csv_macd( series):
    filename = "result_macd.txt"
    file_exists = pd.io.common.file_exists(filename)

    # Extract required values from the series
    values = [
        series.name,  # Datetime
        series['Close'],
        series['macd_line'],
        series['signal_line'],
        series['histogram'],
        series['Position']
    ]

    # Create a new DataFrame with column names
    
    df = pd.DataFrame([values], columns=['Datetime','Close', 'macd_line', 'signal_lines','histogram','Position'])

    # Write the DataFrame to the CSV file
    df.to_csv(filename, mode='a', header=True,sep=" ", index=False)

def append_series_to_csv_bb( series):
    filename = "result_bb.txt"
    file_exists = pd.io.common.file_exists(filename)

    # Extract required values from the series
    values = [
        series.name,  # Datetime
        series['Close'],
        series['upper_band'],
        series['sma'],
        series['lower_band'],
        series['Position']
    ]

    # Create a new DataFrame with column names
    
    df = pd.DataFrame([values], columns=['Datetime','Close', 'upper_band', 'sma','lower_band','Position'])

    # Write the DataFrame to the CSV file
    df.to_csv(filename, mode='a', header=True,sep=" ", index=False)

    
def signal_RSI():
    data = fd.get_live_data('^NSEBANK')
    rsi_data = TA.RSI(data=data, period=14)
    data['RSi'] = rsi_data
    df = data
    for index, row in df.iterrows():
        if row['RSi'] < 30:
            row['Position'] = "LONG"     
            append_series_to_csv_rsi(row)
            
        if row['RSi'] > 70:
            row['Position'] = "SHORT"
            append_series_to_csv_rsi(row) 



def signal_MACD():
    data = fd.get_live_data('^NSEBANK')
    # print(data)
    d = TA.MACD(data=data,fast_period=12,slow_period=26,signal_period=9)
    macd_line = d['macd_line']
    signal_line = d['signal_line']
    histogram = d['histogram']
    index = 0
    for i, row in data.iterrows():
        if macd_line[index] > signal_line[index] and macd_line[index-1] <= signal_line[index-1] and histogram[index] > 0:
            print("row:-",row," i:-",i,"index:-",index," LONG")
            row['Position'] = "LONG"  
            append_series_to_csv_macd(row) 
            
        elif macd_line[index] < signal_line[index] and macd_line[index-1] >= signal_line[index-1] and histogram[index] < 0:
            print("row:-",row," i:-",i,"index:-",index,"Short")  
            row['Position'] = "SHORT"
            append_series_to_csv_macd(row) 
        index +=1

# signal_MACD()
def signal_BB():
    data = fd.get_live_data('^NSEBANK')
    d = TA.BB(data=data,period=10,n_std=1) 
    index = 0
    closing = data['Close']
    upper_band = data['upper_band']
    lower_band = data['lower_band']
    for i, row in data.iterrows():
        if closing[index] > upper_band[index-1] and closing[index-1] <= upper_band[index-1]:
            row['Position'] = "Long"
            append_series_to_csv_bb(row)
        elif closing[index] < lower_band[index-1] and closing[index-1] >= lower_band[index-1]:
            row['Position'] = "Short"
            append_series_to_csv_bb(row)
        index +=1      
    return None



signal_RSI()
signal_MACD()
signal_BB()

# 


