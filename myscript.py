#### LIBRARIES ### 
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from matplotlib import pyplot
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np 
from numpy import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import regularizers
from keras import Sequential
import yfinance as yf

### FUNCTIONS ###
# convert series to supervised learning (credit: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Define the ticker symbols for features and labels
tickers = ["AAPL", "MSFT",'HG=F', 'CL=F', 'NG=F', '^VIX', 'BTC-USD'] #features
target = 'BTC-USD'
target_label = 'Open'

#time period of historical data to fetch
start = "2000-01-01"
end = "2023-09-01"

data_scaling_method = [MinMaxScaler(feature_range=(0, 1)), StandardScaler(), RobustScaler(), QuantileTransformer()]
scaler = data_scaling_method[0]

n_train_days = 365  #size in days of train/test data (i.e. model is tested on last 365 days, and trained on previous days.)
nb_timesteps = 5 #number of timesteps to shift lag (t-3, t-2, t-1, t) for 3 timesteps)

# Fetch historical data for tickers and builds and cleans a Pandas Dataframe
mydf = pd.DataFrame()
for ticker in tickers:
    stock = yf.download(ticker, start, end)
    mydf = pd.concat([mydf, stock['Adj Close']], axis=1, join='outer')
    mydf = pd.concat([mydf, stock['Volume']], axis=1, join='outer')
    stock['Range'] = stock['High'] - stock['Low'] #calculate intraday range
    mydf = pd.concat([mydf, stock['Range']], axis=1, join='outer')
    mydf=mydf.rename(columns={'Adj Close': ticker + " Adj Close", 'Volume' : ticker + ' Volume', 'Range' : ticker + ' Intraday Range'})
label = yf.download(target, start, end)
mydf = pd.concat([mydf, label[target_label]], axis=1, join='outer')
mydf=mydf.rename(columns={target_label: target + ' ' + target_label})
mydf.dropna(inplace=True) #drops days where data is not available for all tickers
mydf = mydf.loc[:, (mydf != 0).any(axis=0)] #drops columns with all zeros
print(mydf.columns) #Prints columns of the data used for training, last column is target

#normalizing and scaling data
dataset = pd.DataFrame(mydf)
nb_features = dataset.shape[1]-1 #(FEATURES --> 'volume', 'open', 'close', etc. However, 7th entry 'price' is a LABEL.)
values = dataset.values #values method give numpy objects instead of df object

# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, nb_timesteps, 1) #(data, number of timesteps t to shift lag, number of timesteps to lead )

# drop columns we don't want to predict
reframed.drop(reframed.columns[[-x for x in range(nb_features,0,-1)]], axis=1, inplace=True) #drops multivariate features from dataset and leaves only 'price' to be forecasted 

### RESHAPE DATA ### 
# split into train and test sets
values = reframed.values
train = values[:-n_train_days, :]
test = values[-n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('\n')
print('The following shapes show the size of the train and test data:')
print('\n')
print('train_X -->    train_y')
print(train_X.shape, train_y.shape)
print('\n')
print('test_X -->   test_y')
print(test_X.shape, test_y.shape)

# DESIGN NEURAL NETWORK MODEL 

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(120, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# To use inverse_transform, you need to reshape lstm_output_scaled to match feature space
# Here we create a dummy array with original feature space

dates_index = pd.bdate_range(end=end, periods = n_train_days)
zeros= (np.squeeze(yhat).shape[0],dataset.values.shape[1]-1)
dummy_input = np.hstack((yhat[:,:,0], np.zeros(zeros))) #note, np.hstack takes TUPLE, and, :,:,0 gives 2D which is needed.

# Reverse the scaling on dummy_input
dummy_input2 = np.hstack((test_y.reshape(-1, 1), np.zeros(zeros))) #note, -1,reshape makes it into 2D which is needed!
original_y = scaler.inverse_transform(dummy_input)[:, 0]  # Extract only the original label
original_test = scaler.inverse_transform(dummy_input2)[:, 0]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the first time series
ax.plot(dates_index, original_test, label=f'Test data {target} {target_label}', color='blue')

# Plot the second time series
ax.plot(dates_index, original_y, label=f'Prediction {target} {target_label}', color='red')

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Values')
ax.set_title(f'Predicted {target_label} trained on data from {start} to {end}')
ax.legend()

rmse = sqrt(mean_squared_error(original_test, original_y))
print('Test RMSE: %.3f' % rmse)
# Show the plot
plt.show()
