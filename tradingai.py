import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getData() -> list:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, batch_size)
    keys = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    dictionary = dict(zip(keys, zip(*rates)))
    data = pd.DataFrame(dictionary)

    print(data['time'])

    #TODO: Fix time from 
    # data['time'] = data['time'].apply(lambda x: (int(x.split(":")[0]) * 60 * 60) + (int(x.split(":")[1]) * 60))

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
    #removing the first 200 entries for they have an NaN for the avg200
    
    data = data[data['avg200'] > 0]
    data = data.reset_index(drop=True)

    ## logic to determine what kind of entry we should make based on the future candles
    ## adding columns bs initializing it with 0.5 "buysell", column will be the 'correct' answer during training and testing
    ## 1 being buy, 0 being sell and 0.5 being do nothing
    # Define the threshold values for your scalp strategy
    # look_ahead_period = 5  # Number of candles to look ahead
    # buy_threshold = 100  # Price increase threshold for a buy signal
    # sell_threshold = -100  # Price decrease threshold for a sell signal

    # # Initialize the 'bs' column with a default value (e.g., 0.5)
    # data['bs'] = 0.5

    # # Iterate over the DataFrame and set the 'bs' column based on future price movement
    # for i in range(len(data) - look_ahead_period):
    #     current_price = data.loc[i, 'close']
    #     future_prices = data.loc[i + 1 : i + look_ahead_period, ['close', 'high', 'low', 'open']]
        
    #     # Check if any future price exceeds the buy threshold
    #     if any((future_prices - current_price).max(axis=1) >= buy_threshold):
    #         data.loc[i, 'bs'] = 1  # Set 'bs' to 1 for a buy signal
        
    #     # Check if any future price meets the sell threshold (for short scenarios)
    #     elif any((future_prices - current_price).min(axis=1) <= sell_threshold):
    #         data.loc[i, 'bs'] = 0  # Set 'bs' to 0 for a sell signal

    
    return data

def getInputData(data) -> list:
    def get_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data[i : i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    # Define input and output data
    X_cols = ['time', 'open', 'high', 'low', 'close', 'real_volume', 'avg200', 'avg50', 'avg22', 'obv', 'vwap']
    X = data[X_cols].values

    # Apply Min-Max scaling to X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sequence_length = 50
    print("Sequence length: ", sequence_length)

    # Generate sequences
    X_inputs = get_sequences(X_scaled, sequence_length)
    return X_inputs
    
# Load the trained model
model_path = './model/best_model.h5'  # Path to the selected model file
model = tf.keras.models.load_model(model_path)

# Connect to MetaTrader5
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

mt5.wait_connected()

# Define symbol and timeframe
symbol = 'WIN$'
timeframe = mt5.TIMEFRAME_M5

# Define prediction batch size
batch_size = 100

# Main loop for continuous prediction and learning
while True:
    data = getData()
    input_data = getInputData(data)
    predictions = model.predict(input_data)
    
    print(predictions)
    # Interpret the predictions and make trading decisions
    # ...
    
    # Update and learn from new data periodically
    # ...
