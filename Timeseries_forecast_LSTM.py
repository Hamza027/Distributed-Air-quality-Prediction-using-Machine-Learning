
# coding: utf-8

# In[1]:

# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:

import requests
from datetime import datetime
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate

from timerseries_forecast_data_preprocess import *



# **Time Series Forecasting Functions**

# In[8]:

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
	    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
	    model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(len(X), 1, 1,)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# Update LSTM model
def update_model(model, train):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    for i in range(1):
         model.fit(X, y, epochs=5, batch_size=1, verbose=0, shuffle=False)
         model.reset_states()
 

#Function to Get live pollutants values from waqi.info for the station
def get_live_pollutants():
  
    # datetime object containing current date and time
    row = datetime.now()
    city_id = '463'
    url = 'https://api.waqi.info/feed/@' + city_id + '/?token='
    api_key = 'f3ff22a8f188bc6d3221eeb2460d8e1882f18408'

    main_url = url + api_key
    r = requests.get(main_url)
    results = r.json()['data']
    #aqi = results['aqi']
    iaqi = results['iaqi']
    # dew = iaqi.get('dew','Nil')
    no2 = iaqi.get('no2','Nil')
    o3 = iaqi.get('o3','Nil')
    so2 = iaqi.get('so2','Nil')
    pm10 = iaqi.get('pm10','Nil')
    pm25 = iaqi.get('pm25','Nil')

    # new_data = pd.DataFrame(columns = ['year','month','day','hour','PM2.5','PM10','SO2','NO2','O3','station'])
    # new_data.loc[0] = [row.year,row.month,row.day,row.hour,pm25['v'],pm10['v'],so2['v']*10,no2['v'],o3['v'],1]
    new_data = pd.DataFrame(columns = ['PM2.5','PM10','SO2','NO2','O3','station'])
    new_data.loc[0] = [pm25['v'],pm10['v'],so2['v']*10,no2['v'],o3['v'],1]

    return new_data


# In[9]:

def train_predict(series, n):
    global lstm_model
    pred_size = -24
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:pred_size], supervised_values[pred_size:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train,test)

    if n==1:
        # fit the model
        lstm_model = fit_lstm(train_scaled, 1,5, 10)

    if n==2:
        # update the model
        update_model(lstm_model,train_scaled)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X = test_scaled[i,0:-1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        #print('Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, raw_values[i+pred_size]))

    # report performance
    
    mae = mean_absolute_error(raw_values[pred_size:], predictions)
    print('Test MAE: %.3f' % mae)
    
    rmse = sqrt(mean_squared_error(raw_values[pred_size:], predictions))
    print('Test RMSE: %.3f' % rmse)

    #line plot of observed vs predicted
    plt.plot(predictions, label='predicted')
    plt.plot(raw_values[pred_size:],label='actual')
    plt.legend()
    plt.show()
    return predictions


# In[10]:

def LSTM_main():
    
    print("LSTM Node ")
    
    #train & Predict Function
    LSTM_Prediction= train_predict(series,1)
    
    return LSTM_Prediction
    
    
    
    
    # In[11]:
    
def inc_learn():    
    #Live data feed in model for incremental learning
    
    prev_data = data[-96:] 
    while True:
        for i in range (1,12):
            new_data = get_live_pollutants()
            print(new_data)
            prev_data.append(new_data)
            
        G_data = prev_data
            
        #Pass raw pollutants and measure 16/8 hours average value
        G_data = group_data(G_data)
            
        #calculate sub_index of each pollutants using 16/8 average value
        G_data = calculate_sub_index(G_data)
            
            
        #calculate AQI from Raw pollutants sub_index value
        G_data = calculate_AQI(G_data)
            
        #Get series of values from data for Time Forecast
        series_1 = G_data["AQI"]
            
        prev_data=G_data
            
        #update & Predict Function
        train_predict(series_1,2)
            
        time.sleep(0)

    
if __name__ == "__main__":
    
    LSTM_res = LSTM_main()
    print("------Incremental Learning LSTM-------")
    inc_learn()

