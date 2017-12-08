#!/usr/bin/env python
import numpy as np
import pandas as pd
from pymongo import MongoClient
from multiprocessing import Pool
import pattern_recognition
import re
import string

# Load auxillary datasets (dollar index, interest spread, volatility index)
dollar = pd.read_csv('dollar.csv')
dollar = dollar.set_index(pd.DatetimeIndex(dollar['DATE']))['value'] / 100

interest = pd.read_csv('interest.csv')
interest = interest.set_index(pd.DatetimeIndex(interest['DATE']))['value']

vol = pd.read_csv('vol.csv')
vol = vol.set_index(pd.DatetimeIndex(vol['DATE']))['value'] / 100

# Load data from MongoDB
client = MongoClient()
collections = client.quandl_stocks.collections
data = [col for col in collections.find()]

def load_data(col, filename='total_data.csv', earliest_date='1990-01-02', volume_threshold=1000000, dollar=dollar, interest=interest, vol=vol, window_length=40, return_window=1, standardize_window=False):

    '''
    Load and format data from MongoDB
    Apply pattern_recognition module, add respective patterns to data along with auxillary data
    Utilize multiprocessing in order to parallelize process
    ---
    IN:
        col: iterable, MongoDB collection/document of unique ticker data
        filename: 'total_data.csv'
        earliest_date: minimum cutoff for stock data
        latest_date: maximum cutoff for stock data
        volume_threshold: minimum average volume a stock must attain in order to added to the final dataset
        dollar, interest, vol: aux datasets
        window_length: lookback price window length, time series input, default=40
        return_window: lookforward (future) price window, target, default=1
        standardize_window: standardize each window by its first price point
    OUT:
        List of each 40-day window along with meta/aux data
    '''

    if not col['Fama Industry'] or not col['Sector'] or col['Exchange'] == 'DELISTED' or col['Fama Industry'] == 'Almost Nothing':
        return

    dates = np.array(col['date']).astype(np.datetime64)
    prices = np.array(col['close'])
    volume = np.array(col['volume'])

    if len(volume) < 150 or len(volume) != len(prices):
        return

    if dates[0] < np.datetime64(earliest_date):
        start_index = np.where(dates == np.datetime64(earliest_date))[0][0]
        dates = dates[start_index:]
        prices = prices[start_index:]
        volume = volume[start_index:]

    ticker = col['Ticker']
    sector = col['Sector']
    industry = col['Fama Industry']
    industry = re.sub('[%s]' % re.escape(string.punctuation), '', industry)

    average_volume = np.mean(volume)

    # Volume threshold
    if average_volume < volume_threshold:

        count = 0
        while average_volume < volume_threshold and count < (len(volume) - 150):
            count += 1
            average_volume = np.mean(volume[count:])

        if average_volume >= volume_threshold:
            dates = dates[count:]
            prices = prices[count:]
            volume = volume[count:]

        else:
            return


    lookback_windows = []

    print(ticker)

    entire_window = window_length + return_window

    for index in range(len(prices) - entire_window + 1):
        sliced_prices = np.array(prices[index : index + entire_window])

        if len(sliced_prices) != entire_window:
            continue

        start_date = dates[index]
        end_date = dates[index + window_length]

        # Auxillary data
        avg_dollar = np.mean(dollar[start_date:end_date])
        avg_interest = np.mean(interest[start_date:end_date])
        avg_vol = np.mean(vol[start_date:end_date])

        # pattern_recognition module
        max_min = pattern_recognition.find_max_min(sliced_prices[:window_length])
        pattern = pattern_recognition.find_patterns(max_min)

        str_start_date = str(start_date)
        str_end_date = str(end_date)
        if not pattern:
            pattern = 'None'
        str_dollar = str(avg_dollar)
        str_interest = str(avg_interest)
        str_vol = str(avg_vol)
        price_string = ''
        for p in sliced_prices:
            price_string += ',' + str(p)

        info_string = ticker + ',' + str_start_date + ',' + str_end_date + ',' + sector + ',' + industry + ',' + pattern + ',' + str_dollar + ',' + str_interest + ',' + str_vol + price_string + '\n'

        lookback_windows.append(info_string)

    return lookback_windows


if __name__=='__main__':
    with open('total_data.csv', 'w') as f:
        pool = Pool(processes=34)
        pool_outputs = pool.map(load_data, data)
        pool.close()
        pool.join()
        # Write list of tickers' windows to one file
        for ticker in pool_outputs:
            if ticker:
                for row in ticker:
                    f.write(row)
