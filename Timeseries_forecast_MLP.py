
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
from numpy import array
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pandas import concat
from math import sqrt
from numpy import concatenate

from timerseries_forecast_data_preprocess import *



# **Time Series Forecasting Functions**

# In[8]:

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# **Capture Live Data from the Station**

# In[9]:

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
    # aqi = results['aqi']
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


# In[10]:

def train_predict(series, n):
    global MLP_model
    pred_size=-24
    n_steps =10
    X, y = split_sequence(series.values, n_steps)
    X_train,Y_train,X_test,Y_test=X[:pred_size], y[:pred_size], X[pred_size:], y[pred_size:]

    if n==1:
        # -------------------------------------
        # setup a MLP network in keras
        # -------------------------------------
        MLP_model = Sequential()
        MLP_model.add(Dense(units = 120, input_dim=n_steps, activation='relu'))
        MLP_model.add(Dropout(0.2))    
        MLP_model.add(Dense(units = 50,  input_dim=n_steps, activation='relu'))      
        MLP_model.add(Dropout(0.2))
        MLP_model.add(Dense(units = 1, activation = 'linear'))
        MLP_model.summary()
        MLP_model.compile(loss='mean_squared_error', optimizer=RMSprop(lr= 0.001))
        MLP_model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=1)
    
    if n==2:
        MLP_model.compile(loss='mean_squared_error', optimizer=RMSprop(lr= 0.0001))
        MLP_model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=0)


    # demonstrate prediction
    yhat = MLP_model.predict(X_test, verbose=0)
    # for i in range(len(Y_test)):
    #     print( 'Predicted=%f, Expected=%f' % (yhat[i], Y_test[i]))

    # report performance
    
    mae = mean_absolute_error(Y_test, yhat)
    print('Test MAE: %.3f' % mae)
    
    rmse = sqrt(mean_squared_error(Y_test, yhat))
    print('Test RMSE: %.3f' % rmse)
    #print(len(raw_values))
    # line plot of observed vs predicted
    plt.plot(yhat, label='predicted')
    plt.plot(Y_test,label='actual')
    plt.legend()
    plt.show()
    return yhat


def MLP_main():
    
    print("MLP Node ")
  
    #train & Predict Function
    MLP_Prediction= train_predict(series,1)
    
    return MLP_Prediction
    
 
def inc_learn(): 
    
     #Live data feed in model for incremental learning
    
     prev_data = data[-96:] 
     while True:
         new_data = get_live_pollutants()
    
         G_data = prev_data.append(new_data)
    
         #Pass raw pollutants and measure 16/8 hours average value
         G_data = group_data(G_data)
    
         #calculate sub_index of each pollutants using 16/8 average value
         G_data = calculate_sub_index(G_data)
    
    
         #calculate AQI from Raw pollutants sub_index value
         G_data = calculate_AQI(G_data)
        
    
         #Get series of values from data for Time Forecast
         series_1 = G_data["AQI"]
        
         prev_data=G_data
    
         # update & Predict Function
         train_predict(series_1,2)
    
         time.sleep(5)

if __name__ == "__main__":
    
    MLP_res = MLP_main()