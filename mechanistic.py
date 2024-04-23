import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

data = pd.read_csv("data/mechanisticdata.csv")

data_past = data[:416]
data_test = data[416:]

#print(data_test)

Cp0 = 158.04 #J/mol-K cellulose and magnesium bisulphite solution
T0 = 333.15 #K
Hfg = 37247.34 #J/mol
Mw = 162.14 #g/mol molar mass of cellulose

# Full test dataset metrics
steam_pred = ((data_test['Sigma'] * Cp0 * T0 * 110 * 1000000 * 0.00001802)/(Hfg * Mw)) + 10.78
absolute_error = np.abs(data_test['CUM_Steam'] - steam_pred)
full_test_mae = absolute_error.mean()
# print("MAE ON TEST DATASET", full_test_mae)

e = abs(steam_pred.mean() - data_test['CUM_Steam'].mean())
print("\nCORRECTION:", e)

# Calculate RMSE for every 5 data points in the test set
batch_size = 5
num_batches = len(data_test) // batch_size
test_rmse_batches = []
test_mae_batches = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_predictions = ((data_test['Sigma'].iloc[start_idx:end_idx] * Cp0 * T0 * data_test['Woodloading'].iloc[start_idx:end_idx] * 1000000 * 0.00001802)/(Hfg * Mw)) + 10.78
    batch_actuals = data_test['CUM_Steam'].iloc[start_idx:end_idx]

    absolute_error = np.abs(batch_actuals - batch_predictions)
    batch_mae = absolute_error.mean()

    batch_rmse = np.sqrt((absolute_error ** 2).mean())

    test_rmse_batches.append(batch_rmse)
    test_mae_batches.append(batch_mae)

# Print RMSE for each batch
print("\nRMSE & MAE FOR EVERY 5 WINDOW")
for i, rmse in enumerate(test_rmse_batches):
    print(f'Batch {i+1} Test RMSE: {rmse:.4f}')

print('\n')

# Print MAE for each batch
for i, mae in enumerate(test_mae_batches):
    print(f'Batch {i+1} Test MAE: {mae:.4f}')

# Average RMSE across batches
avg_rmse = np.mean(test_rmse_batches)
print(f'\nAverage Test RMSE across batches: {avg_rmse:.4f}')

# Average MAE across batches
avg_rmse = np.mean(test_mae_batches)
print(f'\nAverage Test MAE across batches: {avg_rmse:.4f}')

# Plots

plt.plot(data_test['CUM_Steam'], label='Actual')
plt.plot(steam_pred, label='Prediction')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

steam_pred.to_excel("steam_predictions.xlsx", index=False)