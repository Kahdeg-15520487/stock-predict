# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:21:29 2019

@author: kahdeg
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import pandas as pd
import numpy as np
import pandas_datareader.av as pdDataReader
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import datetime
import matplotlib.pyplot as plt
import os

def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
with open('config.json') as configFile:
    config = json.load(configFile)
    apiKey = config['alpha_vantage_api_key']
    
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

#function to get stock data
def aa_stocks(symbol):
    return pdDataReader.time_series.AVTimeSeriesReader(symbol,api_key=apiKey)

def get_historical_stock_price(stock):
    print ("Getting historical stock prices for stock ", stock)
    
    stockData = aa_stocks(stock)
    
    return stockData.read()

def preprocess_stock_data(data):
    print ("Preprocessing data")
    df = data
    df['year'] = data.index.year
    yearList = list(set(df['year'].tolist()))
    dfByYear = []
    for yr in yearList:
        curDf = df[df['year']<=yr]
        dfByYear.append((curDf,yr))
    dfByYear.sort(key=lambda tup:tup[1])
    return dfByYear
#    return 

def main():
    if len(sys.argv) == 2:
        stock = sys.argv[1]
    else:
        stock = input("Enter stock name(ex:GOOGL, AAPL): ")
    createDir('graph/'+stock)
    csvFile ='graph/'+stock+'/data_'+stock+'_'+datetime.datetime.now().strftime("%d_%b_%Y")+'.csv'
    
    if os.path.isfile(csvFile):
        df = pd.read_csv(csvFile,index_col='index')
    else:        
        df_whole = get_historical_stock_price(stock)
        
        df = df_whole.filter(['close'])
            
        df.to_csv(csvFile)
        csvFixed = 'index'
        with open(csvFile, 'r') as myfile:
            csvFixed+=myfile.read()
        with open(csvFile, 'w') as myfile:
            myfile.write(csvFixed)
    
    df.index = pd.to_datetime(df.index)
    df['ds'] = df.index
    #log transform the ‘Close’ variable to convert non-stationary data to stationary.
    df['y'] = np.log(df['close'])
    
    dfByYear = preprocess_stock_data(df)
    firstYear = dfByYear[0][1]
    lastYear = dfByYear[-1][1]
    
    for (dfy,yr) in dfByYear:
        
        initial = (yr-firstYear+1)*365
    
        model = Prophet(
                daily_seasonality=True,
                weekly_seasonality =True,
                yearly_seasonality=True
                )
        
        model.fit(dfy)
        
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        if(yr == lastYear):
            print('initial = '+str(initial))
            df_cv = cross_validation(model, horizon = '180 days')
            print(df_cv.head())
            df_p = performance_metrics(df_cv)
            print(df_p.head())
            fig = plot_cross_validation_metric(df_cv, metric='mape')
            fig.savefig('graph/'+stock+'/'+stock+'_'+str(firstYear)+'_to_'+str(yr)+'_MAPE.svg', bbox_inches='tight', format='svg', dpi=1200)
        
        #Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and
        #the uncertainty intervalsof our forecasts (the blue shaded regions).
        forecast_plot = model.plot(forecast)
#        forecast_plot.show()
        forecast_plot.savefig('graph/'+stock+'/'+stock+'_'+str(firstYear)+'_to_'+str(yr)+'_uncertainty.svg', bbox_inches='tight', format='svg', dpi=1200)
        
        #make the vizualization a little better to understand
        dfy.set_index('ds', inplace=True)
        forecast.set_index('ds', inplace=True)
    #    yearList = list(set(forecast['year'].tolist()))
    #    dfByYear = []
    #    for yr in yearList:
    #        dfByYear.append((forecast[forecast['year']<=yr],df[df['year']<=yr],yr))
    #        
    #    iii = 1+1    
        
    #    for (fy,dfy,yr) in dfByYear:
    
        viz_df = dfy.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
        viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
        
        forecasted = viz_df[['yhat_scaled']].tail(365)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    
        ax1.plot(viz_df.index, viz_df['close'])
        ax1.plot(viz_df.index, viz_df.yhat_scaled, linestyle=':')
        ax1.set_title('Actual Close (Orange) vs Close Forecast (Black)')
        ax1.set_ylabel('Closing Price in Dollars')
        ax1.set_xlabel('Date')
        
        L = ax1.legend() #get the legend
        L.get_texts()[0].set_text('Actual Close')
        L.get_texts()[1].set_text('Forecasted Close')
        
        plt.savefig('graph/'+stock+'/'+stock+'_'+str(yr)+'.svg', bbox_inches='tight', format='svg', dpi=1200)
#        plt.show()
        
        #plot using dataframe's plot function
        viz_df['Actual Close'] = viz_df['close']
        viz_df['Forecasted Close'] = viz_df['yhat_scaled']
        
        viz_df[['Actual Close', 'Forecasted Close']].plot()
    
    print (last10.to_csv())
    
if __name__ == "__main__":
    main()
