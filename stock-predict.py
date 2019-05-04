# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:21:29 2019

@author: kahdeg
"""

import json
import pandas as pd
import numpy as np
import pandas_datareader.av as pdDataReader
from fbprophet import Prophet
import datetime
import matplotlib.pyplot as plt
 
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

def main():
    stock = input("Enter stock name(ex:GOOGL, AAPL): ")

    num_days = int(input("Enter no of days to predict stock price for: "))
    
    # dac diem du lieu, san nao, 
    # cac buoc thuc hien va danh gia model
    df_whole = get_historical_stock_price(stock)
    
    df = df_whole.filter(['close'])
    
    df['ds'] = df.index
    #log transform the ‘Close’ variable to convert non-stationary data to stationary.
    df['y'] = np.log(df['close'])
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=num_days)
    forecast = model.predict(future)
    
    #Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and
    #the uncertainty intervalsof our forecasts (the blue shaded regions).
    forecast_plot = model.plot(forecast)
    forecast_plot.show()
    
    #make the vizualization a little better to understand
    df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    
    viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
    viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(viz_df.index, viz_df['close'])
    ax1.plot(viz_df.index, viz_df.yhat_scaled, linestyle=':')
    ax1.set_title('Actual Close (Orange) vs Close Forecast (Black)')
    ax1.set_ylabel('Closing Price in Dollars')
    ax1.set_xlabel('Date')
    
    L = ax1.legend() #get the legend
    L.get_texts()[0].set_text('Actual Close') #change the legend text for 1st plot
    L.get_texts()[1].set_text('Forecasted Close') #change the legend text for 2nd plot
    
    plt.savefig('graph/prophet_'+stock+'.svg', bbox_inches='tight', format='svg', dpi=1200)
    plt.show()
    
    #plot using dataframe's plot function
    viz_df['Actual Close'] = viz_df['close']
    viz_df['Forecasted Close'] = viz_df['yhat_scaled']
    
    viz_df[['Actual Close', 'Forecasted Close']].plot()
    
    print (forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_days))
    
if __name__ == "__main__":
    main()
