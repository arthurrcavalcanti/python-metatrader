### Author: Arthur Cavalcanti
### 2023-05-01
### Trying to create a model with tensorflow for determining if it should go long or short on a position
### using metatrader5 api for python and assistence from openai chat api

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz;
from keras.models import Sequential
from keras.layers import Dense

def getDataFromMt5() -> list:
    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    # get all symbols
    # symbols=mt5.symbols_get()
    # print('Symbols: ', len(symbols))

    # Get historical data
    symbol = "WIN$"
    timeframe = mt5.TIMEFRAME_M5
    #timezone = pytz.timezone("America/Recife")

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
    
    return data
            

    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()

def processDataDF() -> tuple:
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

data = getDataFromMt5()

X, y = processDataDF()

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Create the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(50, 4)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

model.save('trained_model.h5')