import os
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getData(symbol, timeframe) -> list:
    # Define prediction batch size
    batch_size = 400

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, batch_size)
    keys = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    dictionary = dict(zip(keys, zip(*rates)))
    data = pd.DataFrame(dictionary)
    data.drop('tick_volume', axis=1, inplace=True)
    data.drop('spread', axis=1, inplace=True)

    #fixing the time
    data['time']=pd.to_datetime(data['time'], unit='s')
    data['time'] = (data['time'].dt.hour * 3600) + (data['time'].dt.minute * 60)
    
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
    data = data.tail(51).reset_index(drop=True)
    return data

def getInputData(data, sequence_size = 50) -> list:

    # Define input and output data
    X_cols = data.columns.tolist()
    X = data[X_cols].values

    # Apply Min-Max scaling to X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare the input sequences
    def get_sequences(data, sequence_length):
        sequences = []
        for i in range(sequence_length, len(data)):
            sequence = data[i - sequence_length:i]
            sequences.append(sequence)
        return np.array(sequences)

    sequence_length = sequence_size

    # Generate sequences for training and testing
    sequences = get_sequences(X_scaled, sequence_length)
    return sequences

# function to send a market order
def market_order(symbol, volume, order_type, **kwargs):
    tick = mt5.symbol_info_tick(symbol)

    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    if order_type == 0:
        tp = price_dict[order_type] + 300
        sl = price_dict[order_type] - 100
    else:
        tp = price_dict[order_type] - 300
        sl = price_dict[order_type] + 100

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "deviation": 20,
        "magic": 100,
        "tp": tp,
        "sl": sl,
        "comment": "python market order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order_result = mt5.order_send(request)
    print(order_result)

    return order_result

# function to trail SL
def trail_sl(position, symbol, volume, default_sl = 150, default_sl_distance = 200):
    MAX_DIST_SL = default_sl_distance  # Max distance between current price and SL, otherwise SL will update
    TRAIL_AMOUNT = 55  # Amount by how much SL updates
    DEFAULT_SL = default_sl  # If position has no SL, set a default SL

    MAX_DIST_SL_ABOVE_OPEN_PRICE = 200  # Max distance between current price and SL, otherwise SL will update
    TRAIL_AMOUNT_ABOVE_OPEN_PRICE = 50  # Amount by how much SL updates
    MAX_DIST_SL_ABOVE_OPEN_PRICE_MAX = 100
    # get position data
    order_type = position.type
    price_current = position.price_current
    price_open = position.price_open
    sl = position.sl
    new_sl = sl
    dist_from_sl = abs(round(price_current - sl, 6))
    dist_from_op = abs(round(price_current - price_open, 6))
    if (sl < price_open and order_type == 0) or (sl > price_open and order_type == 1) or sl == 0.0:
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
    elif (sl >= price_open and order_type == 0) or (sl <= price_open and order_type == 1):
        maxDist = MAX_DIST_SL_ABOVE_OPEN_PRICE
        # if dist_from_op > 250:
        #     maxDist = MAX_DIST_SL_ABOVE_OPEN_PRICE_MAX

        if dist_from_sl > maxDist:
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
                "deviation": 20,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            order_result = mt5.order_send(request)
            print(order_result)

            return order_result

    return 'Ticket does not exist'


def withinTime(time):
    return time.minute % TIMEFRAME < TIMEFRAME * 0.4
     
# strategy parameters
SYMBOL = "WINM23"
VOLUME = 1.0
TIMEFRAME = mt5.TIMEFRAME_M5
default_sl = 150
default_sl_distance = 200

flag_threshold = 0.4

# Load the trained model
model_path = './model/model_M5_L0.6020_A0.7940.h5'  # current using a 300 tp and 100 sl training
# model_path = './model/model_M15_L0.5684_A0.8126.h5'  # Path to the selected model file
model = tf.keras.models.load_model(model_path)

# Connect to MetaTrader5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

last_prediction = 0
lastorder = ""
trailorder = ""
trailorderHist = []
orderHist = []
# Main loop for continuous prediction and learning
first_execution = True
signal = f"{bcolors.OKGREEN}Do Nothing{bcolors.ENDC}"
while True:
    current_time = datetime.now()
    if current_time.hour < 9:
        print("Current time is", current_time.strftime("%H:%M:%S"), ", not opperating until after 9 o'clock", " " * 10,  end='\r', flush=True)
        time.sleep(1)
        continue
    elif current_time.hour >= 17:
        print("Its", current_time.strftime("%H:%M:%S"), ", not working till late...")
        print("Closing positions...")
        for pos in mt5.positions_get():
            lastorder = close_order(pos.ticket)
            orderHist.append([lastorder, current_time])

        print("positions closed... bye!")
        break
    first_execution = False
    prev_prediction = last_prediction
    data = getData(SYMBOL, TIMEFRAME)
    input_data = getInputData(data)
    [[buy, sell, nothing]] = model.predict(input_data)
    if buy > nothing and buy > sell and buy > flag_threshold:
        signal = f"{bcolors.OKBLUE}Buy signal{bcolors.ENDC}"
        # if we have a BUY signal, close all short positions
        for pos in mt5.positions_get():
            if pos.type == 1:  # pos.type == 1 represent a sell order
                lastorder = close_order(pos.ticket)
                orderHist.append([lastorder, current_time, signal, buy])
        # if there are no open positions, open a new long position
        if not mt5.positions_total() and withinTime(current_time):
            lastorder = market_order(SYMBOL, VOLUME, 'buy')
            orderHist.append([lastorder, current_time, signal, buy])

    elif sell > nothing and sell > buy and sell > flag_threshold:
        signal = f"{bcolors.OKBLUE}Sell signal{bcolors.ENDC}"
        # if we have a BUY signal, close all short positions
        for pos in mt5.positions_get():
            if pos.type == 0:  # pos.type == 0 represent a buy order
                lastorder = close_order(pos.ticket)
                orderHist.append([lastorder, current_time, signal, sell])

        # if there are no open positions, open a new long position
        if not mt5.positions_total() and withinTime(current_time):
            lastorder = market_order(SYMBOL, VOLUME, 'sell')
            orderHist.append([lastorder, current_time, signal, sell])

    else:
        signal = f"{bcolors.OKGREEN}Do Nothing{bcolors.ENDC}"

    totalPositions = mt5.positions_total()
    # if totalPositions > 0:
    #     for pos in mt5.positions_get():
    #         # tempOrder = trail_sl(pos, SYMBOL, VOLUME, default_sl= default_sl, default_sl_distance=default_sl_distance)
    #         # if tempOrder is not None:
    #         #     trailorder = tempOrder
    #         #     trailorderHist.append([trailorder, current_time])
    #         openprice = pos.price_open
    #         currprice = pos.price_current
    #         if pos.type == 0: #buy position
    #             if currprice - openprice  < -100:
    #                 market_order(SYMBOL, VOLUME, 'sell')
    #         if pos.type == 1: #sold position
    #             if openprice - currprice  < -100:
    #                 market_order(SYMBOL, VOLUME, 'buy')

    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n=====================\n")
    print("Time of last execution:", current_time.strftime("%H:%M:%S"))
    print("Symbol", SYMBOL)
    print("TimeFrame:", TIMEFRAME)
    print("Volume:", VOLUME)
    print(f"{bcolors.WARNING}Flag Threshold set to:", flag_threshold, bcolors.ENDC)
    print("Buy Flag:", buy)
    print("Sell Flag:", sell)
    print("Nothing Flag:", nothing)
    print("Total Positions:", totalPositions)
    print("Current Signal:", signal)
    if totalPositions > 0:
        print("\n   ===== POSITIONS =====")
        for pos in mt5.positions_get():
            if pos.type == 0:
                print("   Position Type: buy")
            if pos.type == 1:
                print("   Position Type: sell")
            
            print("   Position ticket:", pos.ticket)
            print("   Position open price:", pos.price_open)
            print("   Position current price:", pos.price_current)
            print("   Position SL:", pos.sl)
        print("   ===== POSITIONS =====\n")
    if len(orderHist) > 0:
        print("Last order: \n")
        for o in reversed(orderHist[-5:]):
            print("T:", o[1].strftime("%H:%M:%S"),"| OP:", o[0].request.price, "| FlagUsed:", o[2], " - ", o[3], "| C:", o[0].comment)
        print(lastorder)
    
    if len(trailorderHist) > 0:
        print("\nTrail order: \n")
        for t in reversed(trailorderHist[-5:]):
            print("T:", t[1].strftime("%H:%M:%S"),"| Pos:", t[0].request.position, "| SL:", t[0].request.sl,"| C: ", t[0].comment)

        print(trailorder)
    print("\n=====================\n")
    
    time.sleep(1)