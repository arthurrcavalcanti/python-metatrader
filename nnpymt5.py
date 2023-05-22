### Author: Arthur Cavalcanti
### 2023-05-01
### Trying to create a model with tensorflow for determining if it should go long or short on a position
### using metatrader5 api for python and assistence from openai chat api
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz;
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

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

def processDataCsv(data) -> list:
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
    mapper = {"<TIME>": "time", "<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close"}
    data.rename(columns=mapper, inplace=True)
    
    print("original data")
    print(data)
    data = data.drop(['<DATE>', '<TICKVOL>', '<VOL>', '<SPREAD>'], axis=1)
    data['time'] = data['time'].apply(lambda x: (int(x.split(":")[0]) * 60 * 60) + (int(x.split(":")[1]) * 60))
    
    # Calculate moving averages
    data['avg200'] = data['close'].rolling(window=200).mean()
    data['avg50'] = data['close'].rolling(window=50).mean()
    data['avg22'] = data['close'].rolling(window=22).mean()

    # Calculate OBV
    data['obv'] = (data['close'] - data['close'].shift(1)).apply(lambda x: 1 if x > 0 else -1).fillna(0).cumsum()

    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['real_volume']).cumsum() / data['real_volume'].cumsum()

    data.drop('typical_price', axis=1, inplace=True)
    print("new data")
    print(data)
    #removing the first 200 entries for they have an NaN for the avg200
    print("filtering")
    data = data[data['avg200'] > 0]
    data = data.reset_index(drop=True)
    print(data)

    ## logic to determine what kind of entry we should make based on the future candles
    ## adding columns bs initializing it with 0.5 "buysell", column will be the 'correct' answer during training and testing
    ## 1 being buy, 0 being sell and 0.5 being do nothing
    ## to do that i'll set to 1 the rows in which the following 20 candles have at least one value over 200 points from
    ## the current candle, as well as not heving a single candle under by 100 points that way I guarantee a win of 200 
    ## with a risk of 100, 2:1 that is. and set to 0, the other way around. it will remain 0.5 if neither conditions are met
    data['bs'] = 0.5
    print("populating bs column comparing ", len(data), " data")
    for i in range(len(data)):
        currentClose = data.loc[i, 'close']
        futureCandles = data.loc[i+1:]
        # Check if any future candle reaches -200 points below the current candle
        lower_candle = futureCandles[
            (futureCandles['open'] <= currentClose - 200) |
            (futureCandles['high'] <= currentClose - 200) |
            (futureCandles['low'] <= currentClose - 200) |
            (futureCandles['close'] <= currentClose - 200)
        ]

        # Check if any future candle reaches 200 points above the current candle
        upper_candle = futureCandles[
            (futureCandles['open'] >= currentClose + 200) |
            (futureCandles['high'] >= currentClose + 200) |
            (futureCandles['low'] >= currentClose + 200) |
            (futureCandles['close'] >= currentClose + 200)
        ]

        if not lower_candle.empty or not upper_candle.empty:
            if not lower_candle.empty and not upper_candle.empty:
                # Both conditions are met, determine which happened first
                lower_first = lower_candle.index[0] < upper_candle.index[0]
                if lower_first:
                    # Lower condition happened first, set bs to 0
                    data.at[i, 'bs'] = 0
                else:
                    # Upper condition happened first, set bs to 1
                    data.at[i, 'bs'] = 1
            elif not lower_candle.empty:
                # Only lower condition is met, set bs to 0
                data.at[i, 'bs'] = 0
            else:
                # Only upper condition is met, set bs to 1
                data.at[i, 'bs'] = 1
        else:
            # Neither condition is met, set bs to 0.5 (do nothing condition condition)
            data.at[i, 'bs'] = 0.5

    print(data)
    print('saving csv')
    data.to_csv('./bars-data/pre-processed-win$_M5_not_normalized_v2.csv', sep=',')
    return data
    
def processDataCsv3(data) -> list:
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
    
    # Calculate moving averages
    data['avg200'] = data['close'].rolling(window=200).mean()
    data['avg50'] = data['close'].rolling(window=50).mean()
    data['avg22'] = data['close'].rolling(window=22).mean()

    # Calculate OBV
    data['obv'] = (data['close'] - data['close'].shift(1)).apply(lambda x: 1 if x > 0 else -1).fillna(0).cumsum()

    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['real_volume']).cumsum() / data['real_volume'].cumsum()

    data.drop('typical_price', axis=1, inplace=True)
    print("new data")
    print(data)
    #removing the first 200 entries for they have an NaN for the avg200
    print("filtering")
    data = data[data['avg200'] > 0]
    data = data.reset_index(drop=True)
    print(data)

    ## logic to determine what kind of entry we should make based on the future candles
    ## adding columns bs initializing it with 0.5 "buysell", column will be the 'correct' answer during training and testing
    ## 1 being buy, 0 being sell and 0.5 being do nothing
    # Define the threshold values for your scalp strategy
    look_ahead_period = 5  # Number of candles to look ahead
    buy_threshold = 100  # Price increase threshold for a buy signal
    sell_threshold = -100  # Price decrease threshold for a sell signal

    # Initialize the 'bs' column with a default value (e.g., 0.5)
    data['bs'] = 0.5

    # Iterate over the DataFrame and set the 'bs' column based on future price movement
    for i in range(len(data) - look_ahead_period):
        current_price = data.loc[i, 'close']
        future_prices = data.loc[i + 1 : i + look_ahead_period, ['close', 'high', 'low', 'open']]
        
        # Check if any future price exceeds the buy threshold
        if any((future_prices - current_price).max(axis=1) >= buy_threshold):
            data.loc[i, 'bs'] = 1  # Set 'bs' to 1 for a buy signal
        
        # Check if any future price meets the sell threshold (for short scenarios)
        elif any((future_prices - current_price).min(axis=1) <= sell_threshold):
            data.loc[i, 'bs'] = 0  # Set 'bs' to 0 for a sell signal

    print(data)
    print('saving csv')
    data.to_csv('./bars-data/pre-processed-win$_M5_not_normalized_v3.csv', sep=',')
    return data
    

    
def loadPreProcessedDataCsv(pathfile) -> list:
    print("loading data")
    data = pd.read_csv(pathfile, index_col=0)
    print(data)
    return data


def createModel1(data):
    # Preprocess the data
    input_sequences = []
    labels = []

    for i in range(50, len(data)):
        sequence = data.iloc[i-50:i, 1:6].values  # Extract time, open, high, low, close values
        label = data.iloc[i, -1]  # Get the label from 'bs' column
        input_sequences.append(sequence)
        labels.append(label)

    input_sequences = np.array(input_sequences)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_sequences, labels, test_size=0.2, shuffle=False)

    # Build the model
    model = Sequential()
    model.add(LSTM(64, input_shape=(50, 5)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 classes: buy, sell, do nothing

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Make predictions
    predictions = model.predict(X_test)

    print(predictions)

    # Save the model
    print("saving the model")
    model.save('./model/model.h5')

    # # Load the model
    # loaded_model = load_model('model.h5')

def createModel2(data):
    def get_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data[i : i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    # Define input and output data
    X_cols = ['time', 'open', 'high', 'low', 'close', 'avg200', 'avg50', 'avg22']
    y_col = 'bs'
    print("columns defined as x = ", X_cols)
    print("columns defined as y = ", y_col)
    X = data[X_cols].values
    y = data[y_col].values

    # Apply Min-Max scaling to X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("x_scaled len: ", len(X_scaled))
    print(X_scaled)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X_scaled))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    print("split data training x: ", len(X_train), " | y: ", len(y_train))
    # Define the sequence length
    sequence_length = 50
    print("Sequence length: ", sequence_length)

    # Generate sequences
    X_train_sequences = get_sequences(X_train, sequence_length)
    y_train_sequences = y_train[sequence_length:]
    X_test_sequences = get_sequences(X_test, sequence_length)
    y_test_sequences = y_test[sequence_length:]

    print("Seq. Len x: ", len(X_train_sequences), " | y: ", len(y_train_sequences)
          )
    # Create the model
    print("Creating model with 2 LSTM Layers and 2 dense layers")
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length, len(X_cols)), return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    print("compiling model")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("training the model")
    # Train the model
    model.fit(X_train_sequences, y_train_sequences, epochs=10, batch_size=32)

    print("evaluating model")
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_sequences, y_test_sequences)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    
    # Save the model
    print("saving the model")
    model.save('./model/model2_L' + "{:.3f}".format(loss) + '_A' + "{:.3f}".format(accuracy) + '.h5')

def createModel3(data):
    def get_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data[i : i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    # Define input and output data
    X_cols = ['time', 'open', 'high', 'low', 'close', 'real_volume', 'avg200', 'avg50', 'avg22', 'obv', 'vwap']
    y_col = 'bs'
    print("columns defined as x = ", X_cols)
    print("columns defined as y = ", y_col)
    X = data[X_cols].values
    y = data[y_col].values

    # Apply Min-Max scaling to X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("x_scaled len: ", len(X_scaled))
    print(X_scaled)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X_scaled))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    print("split data training x: ", len(X_train), " | y: ", len(y_train))
    # Define the sequence length
    sequence_length = 50
    print("Sequence length: ", sequence_length)

    # Generate sequences
    X_train_sequences = get_sequences(X_train, sequence_length)
    y_train_sequences = y_train[sequence_length:]
    X_test_sequences = get_sequences(X_test, sequence_length)
    y_test_sequences = y_test[sequence_length:]

    print("Seq. Len x: ", len(X_train_sequences), " | y: ", len(y_train_sequences))
    # Create the model
    print("Creating model with 2 LSTM Layers and 2 dense layers")
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length, len(X_cols)), return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    print("compiling model")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #find the next index for the file name
    folder_path = './model'  # Replace with the actual folder path

    # Get a list of files in the folder
    file_list = os.listdir(folder_path)

    # Count the number of files that start with "best_model"
    count = sum(1 for file_name in file_list if file_name.startswith('best_model')) + 1

    # Define the model checkpoint callback
    checkpoint = ModelCheckpoint(filepath='./model/best_model'+str(count)+'.h5', monitor='val_loss', save_best_only=True)

    print("Training the model")
    # Train the model with model checkpointing
    history = model.fit(X_train_sequences, y_train_sequences, epochs=10, batch_size=32,
                        validation_data=(X_test_sequences, y_test_sequences),
                        callbacks=[checkpoint])

    # Load the best model
    best_model = tf.keras.models.load_model('./model/best_model.h5')

    # Evaluate the best model
    loss, accuracy = best_model.evaluate(X_test_sequences, y_test_sequences)
    print("Best Model Loss:", loss)
    print("Best Model Accuracy:", accuracy)

    # print("training the model")
    # # Train the model
    # model.fit(X_train_sequences, y_train_sequences, epochs=20, batch_size=32)

    # print("evaluating model")
    # # Evaluate the model
    # eval_loss, eval_accuracy = model.evaluate(X_test_sequences, y_test_sequences)
    # print("Evaluation Loss:", eval_loss)
    # print("Evaluation Accuracy:", eval_accuracy)

    # print("making predictions")
    # # Make predictions
    # predictions = model.predict(X_test_sequences)

    # # Calculate additional evaluation metrics using predictions and ground truth labels
    # # (e.g., precision, recall, F1 score, etc.)
    # # ...
    # # Convert predictions to binary labels (0 or 1)
    # binary_predictions = (predictions > 0.5).astype(int)

    # # Convert ground truth labels to binary (0 or 1) if needed
    # binary_labels = y_test_sequences.astype(int)

    # # Calculate precision, recall, and F1 score
    # precision = precision_score(binary_labels, binary_predictions)
    # recall = recall_score(binary_labels, binary_predictions)
    # f1 = f1_score(binary_labels, binary_predictions)

    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)

    # # Save the model
    # print("saving the model")
    # model.save('./model/model3_L' + "{:.4f}".format(eval_loss) + '_A' + "{:.4f}".format(eval_accuracy) + '_20epochs.h5')

# data = getDataFromCsv("./bars-data/WIN$_M5_201909041040_202305191800.csv")
# data = processDataCsv3(data)

#loading data
data = loadPreProcessedDataCsv('./bars-data/pre-processed-win$_M5_not_normalized_v3.csv')
createModel3(data)