# Predictive Models of Steam Demand

## Content
1. Description
2. Features
3. Installation
4. Package Requirements
5. Contributions
6. Acknowledgement

## Description
This repository contains the mechanistic model, linear time series model, and neural network models (LSTM, GRU & CNN) used to model and forecast steam demand in a dissolving pulp mill. It also contains the linear and neural network models of the parameter sigma.

## Features
###### 1. Mechanistic Model: The dimensionless parameters that are used as features for the black-box models are in parameter-generation.py. The mechanistic model itself which solves for the mass of steam from parameter sigma is in mechanistic.py.
###### 2. Linear Time Series Model: The ARIMA and ARIMAX models are in the Jupyter Notebook Dig15LinearTSA.ipynb.
###### 3. Neural Network Models: The ANN models are LSTM, GRU & CNN. The univariate models are suffixed by -UV, and the multivariate models are suffixed by -MV.
###### 4. Sigma Model: The ARIMA model of parameter sigma is in the Jupter Notebook sigmaStatisticalAnalysis.ipynb. The ANN models of sigma are in the files SigmapredXXX.py

## Installation
1. Clone the repository to your local machine.
2. Create a virtual environment and activate it.
3. Install the required packages by running 'pip install -r requirements.txt' on the terminal.
4. To run the models on the other digesters, paste the dataset in the data folder and replace the path on the source code.

## Package Requirements
Package requirements for this app can be seen in the requirements.txt file. Install the packages with the command 'pip install -r requirements.txt'

## Contributions
Contributions to the project are welcome. Please submit a pull request with your changes.

## Acknowledgement
Special thanks to Greg Hogg for his invaluable contributions to the neural network time series modelling.

