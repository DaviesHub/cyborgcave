import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Read the CSV file into a pandas DataFrame
df_steam = pd.read_csv("~/Documents/Data/NNRB/finaldata/digester_steam_automate.csv")


def init_func(params, t):
    '''
        Main temperature function
    '''

    delta, Da, sigma = params
    A = (sigma * np.array(t))
    B = delta * Da * (np.exp(-Da*np.array(t) - 1))

    Theta = A - B

    return Theta


def data_converter(temp):
    '''
        This function converts temperature and time to theta and tau respectively and returns them in 
        separate lists
    '''

    temp = temp.tolist()

    tau = [(i*10)/240 for i in range(0, 25)]
    theta = [(t - temp[0])/temp[0] for t in temp]

    return tau, theta


def calculate_parameters_for_batches(df):

    # Initialize a list to store the results for each batch
    batch_results = []
    start_index = 0

    # Iterate through the DataFrame to process batches
    for index, row in df.iterrows():
        steam_flowrate = row['Steam_flowrate']
        temperature = row['Temperature']

        # Check if the current row marks the start of a new batch
        if index + 1 < len(df) and index - 1 >= 0 and df.iloc[index - 1, 1] == 0 and steam_flowrate != 0:
            start_index = index
            end_index = start_index + 24
            cum_steam = df.iloc[end_index, 1]

            # Extract the batch data from the DataFrame
            batch_data = df.iloc[start_index:end_index + 1]

            # Convert to theta tau data
            temp_data = batch_data['Temperature']
            xdata, ydata = data_converter(temp_data)

            # Define the initial guess for the parameters
            p0 = [-0.1, 0.5, 0.05]
            #Correct One p0 = [1, 1, 0.5, 2]

            # Use scipy.optimize.minimize with L-BFGS-B method to optimize the parameters
            res = minimize(lambda p: np.sum((init_func(p, xdata) - ydata)**2), p0, method='L-BFGS-B', tol=1e-6)

            # Get the optimized parameters from the result
            parameters = res.x

            # Store the calculated parameters along with the batch data
            batch_results.append({
                'batch_data': batch_data,
                'parameters': parameters,
                'steam': cum_steam
            })

    return batch_results


def write_parameters_to_csv(results, output_file_path):
    # Prepare the data to be written to the CSV file
    data = []
    for batch_index, batch_result in enumerate(results):
        parameters = batch_result['parameters']
        steam = batch_result['steam']
        data.append([batch_index] + parameters.tolist() + [steam])

    output_file_path = os.path.expanduser(output_file_path)

    # Write data to CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Batch Index', 'Delta', 'Damkohler', 'Sigma', 'CUM_Steam'])  # Header row
        writer.writerows(data)


output_file_path = "~/Documents/Data/NNRB/Finaldata/batch_parameters.csv"
results = calculate_parameters_for_batches(df_steam)

write_parameters_to_csv(results, output_file_path)