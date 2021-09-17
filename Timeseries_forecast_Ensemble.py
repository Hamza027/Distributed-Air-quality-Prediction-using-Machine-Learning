
# coding: utf-8

# In[2]:

# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:

import requests
from datetime import datetime
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from numpy import array
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate






from  timerseries_forecast_data_preprocess import *
from  Timeseries_forecast_LSTM import *
from  Timeseries_forecast_CNN import *
from  Timeseries_forecast_MLP import *
from  Timeseries_forecast_SVM import *

LSTM_P = np.rint(LSTM_res).tolist()

CNN_P = np.rint(CNN_res).ravel()

MLP_P = np.rint(MLP_res).ravel()

SVM_P = np.rint(SVR_res).tolist()

series = series[-24:]

Results = pd.DataFrame()

Results["Actual"] = series
Results["LSTM"] = LSTM_P
Results["CNN"] = CNN_P 
Results["MLP"] = MLP_P
Results["SVR"] = SVM_P
Results["OPM_Result"] = np.rint((Results["LSTM"]*0.40)+(Results["CNN"]*0.10)+(Results["MLP"]*0.10)+(Results["SVR"]*0.40))

print(Results)


plt.figure(figsize= [10,7])
plt.plot(range(1,25),Results["Actual"], label='Actual')
plt.plot(range(1,25),Results["LSTM"], label='LSTM')
plt.plot(range(1,25),Results["CNN"], label='CNN')
plt.plot(range(1,25),Results["MLP"], label='MLP')
plt.plot(range(1,25),Results["SVR"], label='SVR')
plt.plot(range(1,25),Results["OPM_Result"], label='Optimized')
plt.xlabel("Hours")
plt.ylabel("AQI")
plt.legend()
plt.show()

plt.bar(range(1,25),Results)
plt.show


plt.figure(figsize= [10,7])
plt.plot(range(1,25),Results["Actual"], label='Actual')
plt.plot(range(1,25),Results["OPM_Result"], label='Optimized')
plt.xlabel("Hours")
plt.ylabel("AQI")
plt.legend()
plt.show()

rmse = sqrt(mean_squared_error(Results["Actual"], Results["OPM_Result"]))
print('Test RMSE: %.3f' % rmse)

mae = mean_absolute_error(Results["Actual"], Results["OPM_Result"])
print('Test MAE: %.3f' % mae)
#pd.set_option('display.max_columns', 500)
#pd.set_option('expand_frame_repr', False)



