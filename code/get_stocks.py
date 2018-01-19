from pymongo import MongoClient
import urllib
import json
import quandl

token = 'authtoken here'
%matplotlib inline

client = MongoClient()
cols = client.quandl_stocks.collections

with urllib.request.urlopen('http://www.sharadar.com/meta/tickers.json') as url:
    data = json.loads(url.read().decode())
    cols.insert_many(data)
    
attrs = {'Adj. Open': 'open', 'Adj. High': 'high', 'Adj. Low': 'low', 'Adj. Close': 'close', 'Adj. Volume': 'volume'}
for col in cols.find():
    ticker = col['Ticker']
    try:
        stock_data = quandl.get("WIKI/%s" % ticker, authtoken=token)
        cols.update_one({'_id': col['_id']}, {'$set': {'date': stock_data.index.astype(str).tolist()}})
        for key, value in attrs.items():
            cols.update_one({'_id': col['_id']}, {'$set': {value: stock_data[key].tolist()}})
        print(ticker, 'added')
    except:
        cols.remove({'_id': col['_id']})
        print(ticker, 'deleted')