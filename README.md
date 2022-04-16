# Distributed-Air-quality-Prediction-using-Machine-Learning

Machine Learning Project

## The diagram shows the main architecture of the project

![image](https://user-images.githubusercontent.com/26603682/163656600-3e665307-b60d-4216-88d5-f9da65787447.png)


Files Description :

File 1: Timerseries_forecast_data_preprocess.py: Air quality index calculation using raw pollutants, data cleaning, feature engineering and return time series AQI data.


## The diagram illustrates the method to calculate AQI from air pollutants values.
![image](https://user-images.githubusercontent.com/26603682/163656593-8c077f67-699d-4b28-9dd2-2df476593356.png)


File 2: Timeseries_forecast_CNN.py : 1D Convolutional Neural Network model to predict Air Quality index from the time series AQI data.

File 3: Timeseries_forecast_LSTM.py : Long-short Term Memory (Recursive Neural Network) model to predict Air Quality index from the time series AQI data.

File 4: Timeseries_forecast_MLP.py : Multi-Layer Perceptron model to predict Air Quality index from the time series AQI data.

File 5: Timeseries_forecast_SVM.py : Support Vector Regressor model to predict Air Quality index from the time series AQI data.

File 6: Timeseries_forecast_Ensemble.py : Weighted Avergae (Aggregated) result from standalone models based on performance

## The aggregation methodolgy
![image](https://user-images.githubusercontent.com/26603682/163656583-1f657956-e3f9-4ddf-8cce-1210ec1cab06.png)




## The prediction performance of the standalone models, aggreagted model and the real output

![image](https://user-images.githubusercontent.com/26603682/163656572-8955ccb0-e6a8-4b1b-ae09-354f4853e927.png)

## The RMSE and MAE scores of all the models

![image](https://user-images.githubusercontent.com/26603682/163656652-a6eb10af-8564-437f-8d2f-7032aa4d7da3.png)



