import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns

def getDataFromMt5() -> list:
    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    # Get historical data
    symbol = "WIN$"
    timeframe = mt5.TIMEFRAME_M5

    #5min: start_date = 05/09/2019
    #5 min: start_date = 21/05/2018
    #15 min: start_date = 21/05/2018
    start_date = datetime(2019, 9, 5)
    end_date = datetime(2023, 5, 19)

    # Loop through each month
    data = []
    current_date = start_date
    while current_date <= end_date:
        # Set the start and end times for the current month
        start_time = current_date
        end_time = current_date + timedelta(days=30)

        # Retrieve the data for the current month
        temp = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        # Process the data for the current month
        if temp is not None and len(temp)>0:
            data.extend(temp)
            print("SD: " + str(start_time) + " - ED: " + str(end_time) + " | Len: " + str(len(data)))
            # Add your data processing logic here
            # For example, you can calculate indicators or perform analysis
            # Move to the next month
            current_date += timedelta(days=31)
        else:
            print('Temp is none for date: ' + str(current_date))

    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
    
    return data
            
def processDataDF(data) -> tuple:
    keys = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    dictionary = dict(zip(keys, zip(*data)))
    df = pd.DataFrame(dictionary)

    # Normalize the data
    print(df)
    df['time']=pd.to_datetime(df['time'], unit='s')
    df['open'] = df['open'] / df['open'].max()
    df['high'] = df['high'] / df['high'].max()
    df['low'] = df['low'] / df['low'].max()
    df['close'] = df['close'] / df['close'].max()
    print(df)

    print("rearanging data")
    window_size = 50
    total = range(window_size, len(df))
    X = []
    for i in range(window_size, len(df)):
        if i % 40000 == 0:
            print("Total: ", str(len(df)), "curr: ", str(i))
        X.append(df[['open', 'high', 'low', 'close']][i-window_size:i].values)

    print("data rearanged")
    X = np.array(X)
    y = df['close'].shift(-window_size).fillna(0).values.reshape(-1, 1)
    y = y[:len(X)] #this ensures y has the same size as x
    # # Define input and output data
    # X = df[['open', 'high', 'low', 'close']].values
    # y = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 if next close > current close, otherwise 0
    return (X, y)

def getDataFromCsv(filepath) -> list:
    print("getting data from csv")
    data = pd.read_csv(filepath, sep="\t")
    return data

def processDataCsv(data, min) -> list:
    print('processing data')
    # <DATE>       2019.09.04
    # <TIME>         10:40:00
    # <OPEN>           127039
    # <HIGH>           127302
    # <LOW>            126876
    # <CLOSE>          127001
    # <TICKVOL>         38504
    # <VOL>            138900
    # <SPREAD>              1
    ## formating time to be in seconds
    print("renaming columns")
    mapper = {"<TIME>": "time", "<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close", "<VOL>": "real_volume"}
    data.rename(columns=mapper, inplace=True)
    
    print("original data")
    print(data)
    data = data.drop(['<DATE>', '<TICKVOL>', '<SPREAD>'], axis=1)
    data['time'] = data['time'].apply(lambda x: (int(x.split(":")[0]) * 60 * 60) + (int(x.split(":")[1]) * 60))

    # Calculate OBV
    data['obv'] = (data['close'] - data['close'].shift(1)).apply(lambda x: 1 if x > 0 else -1).fillna(0).cumsum()

    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['real_volume']).cumsum() / data['real_volume'].cumsum()

    data.drop('typical_price', axis=1, inplace=True)

    # Calculate Simple Moving Average (SMA)
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()

    # Calculate Exponential Moving Average (EMA)
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()

    # Calculate Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Calculate Moving Average Convergence Divergence (MACD)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    
    # Bollinger Bands
    window = 20
    sma = data['close'].rolling(window).mean()
    std = data['close'].rolling(window).std()
    data['upper_band'] = sma + 2 * std
    data['lower_band'] = sma - 2 * std

    # Stochastic Oscillator
    window = 14
    lowest_low = data['low'].rolling(window).min()
    highest_high = data['high'].rolling(window).max()
    data['stochastic'] = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100

    # Drop any rows with missing values
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    print("new data after the addition of indicators")
    print(data)

    ## logic to determine what kind of entry we should make based on the future candles
    # Define the threshold values for your scalp strategy
    look_ahead_period = 5  # Number of candles to look ahead
    buy_threshold = 300  # Price increase threshold for a buy signal
    sell_threshold = -300  # Price decrease threshold for a sell signal
    buy_sl_threshold = -100
    sell_sl_threshold = 100
    data['buy'] = 0
    data['sell'] = 0
    data['nothing'] = 1
    # Iterate over the DataFrame and set the 'bs' column based on future price movement
    counter_of_conditions_met = 0
    for i in range(len(data) - look_ahead_period):
        
        current_candles = data.loc[i, 'close']
        future_candles = data.loc[i + 1:i + look_ahead_period, ['open', 'high', 'low', 'close']]

        # Check conditions for buy and sell thresholds
        diff = future_candles - current_candles
        stacked_diff = diff.stack()
        max_index = stacked_diff.idxmax()
        max_value = stacked_diff.loc[max_index]
        candles_up_to_max = stacked_diff[:max_index]
        min_index_up_to_max = candles_up_to_max.idxmin()
        min_value_up_to_max = candles_up_to_max.loc[min_index_up_to_max[0]].min()
        min_index = stacked_diff.idxmin()
        min_value = stacked_diff.loc[min_index]
        candles_up_to_min = stacked_diff[:min_index]
        max_index_up_to_min = candles_up_to_min.idxmax()
        max_value_up_to_min = candles_up_to_min.loc[max_index_up_to_min[0]].max()
        
        buy_condition_met = max_value > buy_threshold and min_value_up_to_max < buy_sl_threshold
        sell_condition_met = min_value < sell_threshold and max_value_up_to_min > sell_sl_threshold
        
        print("Index:", i, "conditions met:", counter_of_conditions_met," " * 20, end='\r', flush=True)
        
        if buy_condition_met and sell_condition_met:
            first_buy_index = max_index[0]
            first_sell_index = min_index[0]
            counter_of_conditions_met += 1
            if first_buy_index < first_sell_index:
                data.at[i, 'buy'] = 1
                data.at[i, 'nothing'] = 0
            else:
                data.at[i, 'sell'] = 1
                data.at[i, 'nothing'] = 0
        elif buy_condition_met:
            data.at[i, 'buy'] = 1
            data.at[i, 'nothing'] = 0
            counter_of_conditions_met += 1
        elif sell_condition_met:
            data.at[i, 'sell'] = 1
            data.at[i, 'nothing'] = 0
            counter_of_conditions_met += 1
            

    
    print("Final data: ")
    print(data)
    print('saving csv')
    folder_path = './bars-data/'
    file_list = os.listdir(folder_path)
    
    count = sum(1 for file_name in file_list if file_name.startswith('pre-processed-win$_M' + str(min)))
    filename = 'pre-processed-win$_M' + str(min) + '_not_normalized_v'+str(count)+'.csv'
    csvPath = os.path.join(folder_path, filename)
    data.to_csv(csvPath, sep=',')

    return data
    
data = getDataFromCsv("./bars-data/WIN$_M5_201909041040_202305191800.csv")
data = processDataCsv(data, 5)

# data = getDataFromCsv("./bars-data/WIN$_M15_201805230900_202305231030.csv")
# data = processDataCsv(data, 15)
