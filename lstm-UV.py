import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.metrics import MeanAbsoluteError
#import xlsxwriter

# Load the dataset from the Excel CSV file
csv_file_path = "data/NNdataset.csv"
df = pd.read_csv(csv_file_path)

steam = df['CUM_Steam']

# Converting forecasting problem to supervised learning problem
def df_to_X_y(df, window):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df) - window):
        row = [[a] for a in df_as_np[i:i+window]]
        X.append(row)
        label = df_as_np[i+window]
        y.append(label)

    return np.array(X), np.array(y)

#print(len(df))

WINDOW_SIZE = 3
X, y = df_to_X_y(steam, WINDOW_SIZE)

# Split into train, test, validation

X_train, y_train = X[:370], y[:370]
X_val, y_val = X[370:416], y[370:416]
X_test, y_test = X[416:], y[416:]

X_test = X_test
y_test = y_test

# print(X_train.shape,  y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# Compile model
model = Sequential()
model.add(InputLayer((3, 1)))
model.add(LSTM(64))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

print(model.summary())

cp = ModelCheckpoint('model/', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

model.fit(X_train, tf.convert_to_tensor(y_train), validation_data=(X_val, tf.convert_to_tensor(y_val)), epochs=95, callbacks=[cp])

# Predictions for all datasets
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

# Calculate RMSE for each dataset using RootMeanSquaredError metric
train_rmse = RootMeanSquaredError()(y_train, train_predictions).numpy()
val_rmse = RootMeanSquaredError()(y_val, val_predictions).numpy()
# test_rmse = RootMeanSquaredError()(y_test, test_predictions).numpy()

print('Training RMSE:', train_rmse)
print('Validation RMSE:', val_rmse)
# print('Test RMSE:', test_rmse)

# Calculate MAE for each dataset
train_mae = MeanAbsoluteError()(y_train, train_predictions).numpy()
val_mae = MeanAbsoluteError()(y_val, val_predictions).numpy()
# test_mae = MeanAbsoluteError()(y_test, test_predictions).numpy()

print('\nTraining MAE:', train_mae)
print('Validation MAE:', val_mae)
# print('Test MAE:', test_mae)

# Calculate RMSE for every 5 data points in the test set
batch_size = 5
num_batches = len(y_test) // batch_size
test_rmse_batches = []
test_mae_batches = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_predictions = model.predict(X_test[start_idx:end_idx]).flatten()
    batch_actuals = y_test[start_idx:end_idx]

    batch_rmse = RootMeanSquaredError()(batch_actuals, batch_predictions).numpy()
    batch_mae = MeanAbsoluteError()(batch_actuals, batch_predictions).numpy()

    test_rmse_batches.append(batch_rmse)
    test_mae_batches.append(batch_mae)

# Print RMSE for each batch
for i, rmse in enumerate(test_rmse_batches):
    print(f'Batch {i+1} Test RMSE: {rmse:.4f}')

print('\n')

# Print RMSE for each batch
for i, mae in enumerate(test_mae_batches):
    print(f'Batch {i+1} Test MAE: {mae:.4f}')

# Average RMSE across batches
avg_rmse = np.mean(test_rmse_batches)
print(f'\nAverage Test RMSE across batches: {avg_rmse:.4f}')

# Average MAE across batches
avg_rmse = np.mean(test_mae_batches)
print(f'\nAverage Test MAE across batches: {avg_rmse:.4f}')

# Plot predictions vs. actuals for training, validation, and test datasets
# Plot the training predictions and actuals
plt.figure(figsize=(9, 5))
plt.plot(y_train, label='Actual Training Data', marker='x', markersize=3, linestyle='-', linewidth=1, color='blue')
plt.plot(train_predictions, label='Predictions', marker='o', markersize=3, linestyle='-', linewidth=1, color='red')
plt.xlabel('Batch Number')
plt.ylabel('Steam Demand (tonne)')
plt.title('LSTM-UV(3) Model Predictions vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()

# Validation set
plt.figure(figsize=(9, 5))
plt.plot(y_val, label='Actuals', marker='x', markersize=3, linestyle='-', color='blue')
plt.plot(val_predictions, label='Predictions', marker='o', markersize=3, linestyle='-', color='red')
plt.title('LSTM-UV(3) Validation Predictions vs Actuals')
plt.xlabel('Batch Number')
plt.ylabel('Steam Demand (tonne)')
plt.grid(True)
plt.legend()
plt.show()

# Test set
plt.figure(figsize=(9, 5))
plt.plot(y_test, label='Actual Test Data', marker='x', markersize=3, linestyle='-', color='blue')
plt.plot(test_predictions, label='Forecasts', marker='o', markersize=3, linestyle='-', color='red')
plt.title('LSTM-UV(3) Model Forecast vs Actual Data')
plt.xlabel('Batch Number')
plt.ylabel('Steam Demand (tonne)')
plt.grid(True)
plt.legend()
plt.show()


# Create a new Excel file and add a worksheet.
# workbook = xlsxwriter.Workbook('lstmsv_training_and_testing_predictions.xlsx')
# worksheet = workbook.add_worksheet()

# # Headers
# worksheet.write('A1', 'y_train')
# worksheet.write('B1', 'train_predictions')
# worksheet.write('C1', 'y_test')
# worksheet.write('D1', 'test_predictions')

# # Function to write data to a column
# def write_column(sheet, col, data, row_start=1):
#     for i, value in enumerate(data):
#         sheet.write(i + row_start, col, float(value))

# # Write data to Excel
# write_column(worksheet, 0, y_train.flatten())
# write_column(worksheet, 1, train_predictions)
# write_column(worksheet, 2, y_test.flatten())
# write_column(worksheet, 3, test_predictions)

# # Close the workbook
# workbook.close()
