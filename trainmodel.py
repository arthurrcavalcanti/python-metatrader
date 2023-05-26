import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras import optimizers

# Load your dataset into a pandas DataFrame
min = '5'
folder_path = './bars-data/'
file_list = os.listdir(folder_path)
count = sum(1 for file_name in file_list if file_name.startswith('pre-processed-win$_M' + min)) - 1
file_name = 'pre-processed-win$_M' + min + '_not_normalized_v' + str(count) + '.csv'
file_path = os.path.join(folder_path, file_name)
print("Data used:", file_path)
data = pd.read_csv(file_path, index_col=0)

# Define your input columns and target column
X_cols = data.drop(['buy', 'sell', 'nothing'], axis=1).columns.tolist()
print("X_cols:", X_cols)
y_cols = ['buy', 'sell', 'nothing']
print("y_cols:", y_cols)

# Extract the input features (X) and target variable (y)
X = data[X_cols].values
y = data[y_cols].values

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define a function to generate sequences from the data
def get_sequences(data, targets, sequence_length):
    sequences = []
    sequence_targets = []
    for i in range(sequence_length, len(data)):
        sequence = data[i - sequence_length:i]
        sequence_target = targets[i]  # Get the corresponding target value
        sequences.append(sequence)
        sequence_targets.append(sequence_target)
    return np.array(sequences), np.array(sequence_targets)

sequence_length = 50

# Generate sequences for training and testing
X_train_sequences, y_train_sequences = get_sequences(X_scaled, y, sequence_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_sequences, y_train_sequences, test_size=0.2, shuffle=False)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, len(X_cols)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# learning_rate = 0.01
# optimizer = optimizers.Adam(learning_rate=learning_rate)
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy', 'mae', 'mse']
print("M:", min, "| Optimizer:", optimizer, "| Loss:", loss, "| Metrics:", metrics)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Define the model checkpoint callback
folder_path = './model/'
file_list = os.listdir(folder_path)
count = sum(1 for file_name in file_list if file_name.startswith('best_model'))
filename = 'best_model'+str(count)+'.h5'
modelPath = os.path.join(folder_path, filename)
print("Model path:", modelPath)

checkpoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True)
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), 
                    callbacks=[checkpoint])

# Evaluate the model on the test set
loss, accuracy, mae, mse = model.evaluate(X_test, y_test)
print("Model Loss:", loss)
print("Model Accuracy:", accuracy)
print("Model mae:", mae)
print("Model mse:", mse)

#renaming model
newFileName = 'model_M' + min + '_L' + "{:.4f}".format(loss) + '_A' + "{:.4f}".format(accuracy) + '.h5'
newPath = os.path.join(folder_path, newFileName)
os.rename(modelPath, newPath)
print("Model renamed:", newFileName)
print("Saved at:", newPath)