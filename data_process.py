# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:01:44 2019

@author: kahdeg
"""

import os
import sys
import requests
import json
import csv
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime

AUTO_CLEANUP = False
FILE_NAME_JSON = 'historical.json'
FILE_NAME_CSV = 'historical.csv'

with open('config.json') as configFile:
    config = json.load(configFile)
    apiKey_aa = config['alpha_vantage_api_key']
    apiKey_pl = config['plotly_api_key']
    

def get_historical(symbol):
    # Download our file from google finance
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&apikey='+apiKey_aa+'&outputsize=full'
    print(url)
    r = requests.get(url, stream=True)
    print(r.status_code)
    if r.status_code != 400:
        with open(FILE_NAME_JSON, 'wb') as f:
            for chunk in r:
                f.write(chunk)

        return True
    
def plot():
    plotly.tools.set_credentials_file(username='kahdeg', api_key=apiKey_pl)

    df = pd.read_csv('historical.csv')
    
    trace = go.Ohlc(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])
    data = [trace]
    py.iplot(data, filename='simple_ohlc')
    
symbol = input('enter a valid stock symbol: ').upper()

# Check if we got the historical data
if not get_historical(symbol):
    print ('enter a valid stock symbol: ')
    sys.exit()

with open(FILE_NAME_JSON) as f:
    jsondata = json.load(f)
    
dataKeys = list(jsondata.keys())

data = jsondata[dataKeys[1]]

dataDates = list(data)

print(dataDates)

    
with open(FILE_NAME_CSV, mode='w') as csv_file:
    fieldnames = ['date', 'open', 'high', 'low', 'close']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for date in dataDates:
        row = data[date]
        
        print(date+":")
        print(data[date])
        writer.writerow({'date': date, 'open': row['1. open'], 'high': row['2. high'], 'low': row['3. low'], 'close': row['4. close']})
    
plot()