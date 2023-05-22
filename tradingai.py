import os
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def getData(symbol, timeframe) -> list:
    # Define prediction batch size
    batch_size = 400

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, batch_size)
    keys = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    dictionary = dict(zip(keys, zip(*rates)))
    data = pd.DataFrame(dictionary)

    #fixing the time
    data['time']=pd.to_datetime(data['time'], unit='s')
    data['time'] = (data['time'].dt.hour * 3600) + (data['time'].dt.minute * 60)
    
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
    data = data.tail(51).reset_index(drop=True)
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

    # Generate sequences
    X_inputs = get_sequences(X_scaled, sequence_length)
    return X_inputs

# function to send a market order
def market_order(symbol, volume, order_type, **kwargs):
    tick = mt5.symbol_info_tick(symbol)

    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "deviation": 20,
        "magic": 100,
        "comment": "python market order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order_result = mt5.order_send(request)
    print(order_result)

    return order_result

# function to trail SL
def trail_sl(position, symbol, volume):
    MAX_DIST_SL = 200  # Max distance between current price and SL, otherwise SL will update
    TRAIL_AMOUNT = 52  # Amount by how much SL updates
    DEFAULT_SL = 150  # If position has no SL, set a default SL

    MAX_DIST_SL_ABOVE_OPEN_PRICE = 150  # Max distance between current price and SL, otherwise SL will update
    TRAIL_AMOUNT_ABOVE_OPEN_PRICE = 75  # Amount by how much SL updates

    # get position data
    order_type = position.type
    price_current = position.price_current
    price_open = position.price_open
    sl = position.sl
    new_sl = sl
    dist_from_sl = abs(round(price_current - sl, 6))

    if(sl < price_open):
        if dist_from_sl > MAX_DIST_SL:
            # calculating new sl
            if sl != 0.0:
                if order_type == 0:  # 0 stands for BUY
                    new_sl = sl + TRAIL_AMOUNT

                elif order_type == 1:  # 1 stands for SELL
                    new_sl = sl - TRAIL_AMOUNT

            else:
                # setting default SL if the is no SL on the symbol
                new_sl = price_open - DEFAULT_SL if order_type == 0 else price_open + DEFAULT_SL 
    elif sl >= price_open:
        if dist_from_sl > MAX_DIST_SL_ABOVE_OPEN_PRICE:
            # calculating new sl
            if order_type == 0:  # 0 stands for BUY
                new_sl = sl + TRAIL_AMOUNT_ABOVE_OPEN_PRICE

            elif order_type == 1:  # 1 stands for SELL
                new_sl = sl - TRAIL_AMOUNT_ABOVE_OPEN_PRICE

    if new_sl != sl:
        request = {
                "symbol": symbol,
                "volume": volume,
                "action": mt5.TRADE_ACTION_SLTP,
                "deviation": 20,
                "magic": 100,
                "comment": "python TRADE_ACTION_SLTP",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "sl": new_sl,
                "position": position.ticket
            }
        
        result = mt5.order_send(request)
        return result




# function to close an order base don ticket id
def close_order(ticket):
    positions = mt5.positions_get()

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        type_dict = {0: 1, 1: 0}  # 0 represents buy, 1 represents sell - inverting order_type to close the position
        price_dict = {0: tick.ask, 1: tick.bid}

        if pos.ticket == ticket:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": DEVIATION,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            order_result = mt5.order_send(request)
            print(order_result)

            return order_result

    return 'Ticket does not exist'


# function to get the exposure of a symbol
def get_exposure(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        pos_df = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
        exposure = pos_df['volume'].sum()

        return exposure

# Load the trained model
model_path = './model/best_model2.h5'  # Path to the selected model file
model = tf.keras.models.load_model(model_path)

# Connect to MetaTrader5
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# strategy parameters
SYMBOL = "WINM23"
VOLUME = 1.0
TIMEFRAME = mt5.TIMEFRAME_M5
last_prediction = 0
lastorder = ""
trailorder = ""
# Main loop for continuous prediction and learning
while True:
    prev_prediction = last_prediction
    data = getData(SYMBOL, TIMEFRAME)
    input_data = getInputData(data)
    last_prediction = model.predict(input_data)[0]
    dist_prediction = last_prediction - prev_prediction
    threshold = 0.75
    signal = "No action"
    totalPositions = mt5.positions_total()
    if last_prediction > threshold:
        # if we have a BUY signal, close all short positions
        for pos in mt5.positions_get():
            if pos.type == 1:  # pos.type == 1 represent a sell order
                lastorder = close_order(pos.ticket)

        # if there are no open positions, open a new long position
        if not mt5.positions_total():
            lastorder = market_order(SYMBOL, VOLUME, 'buy')

        signal = "Buy signal"
    elif last_prediction < (1 - threshold):
        # if we have a BUY signal, close all short positions
        for pos in mt5.positions_get():
            if pos.type == 0:  # pos.type == 0 represent a buy order
                lastorder = close_order(pos.ticket)

        # if there are no open positions, open a new long position
        if not mt5.positions_total():
            market_order(SYMBOL, VOLUME, 'sell')
        signal = "Sell signal"
    else:
        signal = "Do Nothing"

    if totalPositions > 0:
        for pos in mt5.positions_get():
            tempOrder = trail_sl(pos, SYMBOL, VOLUME)
            if tempOrder is not None:
                trailorder = tempOrder
            # openprice = pos.price_open
            # currprice = pos.price_current
            # if pos.type == 0: #buy position
            #     if currprice - openprice  < -100:
            #         market_order(SYMBOL, VOLUME, 'sell')
            # if pos.type == 1: #sold position
            #     if openprice - currprice  < -100:
            #         market_order(SYMBOL, VOLUME, 'buy')

    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n=====================\n")
    print("Symbol ", SYMBOL)
    print("TimeFrame: ", TIMEFRAME)
    print("Volume: ", VOLUME)
    print("Last Prediction: ", last_prediction)
    print("Prev Prediction: ", prev_prediction)
    print("Dist Prediction: ", dist_prediction)
    print("Total Positions: ", totalPositions)
    print("Current Signal: ", signal)
    if totalPositions > 0:
        print("===== POSITIONS =====")
        for pos in mt5.positions_get():
            if pos.type == 0:
                print(" Position Type: bought")
            if pos.type == 1:
                print(" Position Type: sold")
            
            print(" Position ticket: ", pos.ticket)
            print(" Position open price: ", pos.price_open)
            print(" Position current price: ", pos.price_current)
            print(" Position SL: ", pos.sl)
        print("===== POSITIONS =====")
    print("Last order: \n")
    print(lastorder)
    print("Trail order: \n")
    print(trailorder)
    print("\n=====================\n")

    time.sleep(1)
    
    # Interpret the predictions and make trading decisions
    # ...
    
    # Update and learn from new data periodically
    # ...
