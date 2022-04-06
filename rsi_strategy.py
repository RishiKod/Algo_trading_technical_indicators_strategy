import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import copy

def rsi(df, n = 14):
    df.dropna(inplace = True)
    df['gain'] = np.where(df['Adj Close']>df['Adj Close'].shift(1), df['Adj Close']-df['Adj Close'].shift(1), 0)
    df['loss'] = np.where(df['Adj Close']<df['Adj Close'].shift(1), abs(df['Adj Close'].shift(1)-df['Adj Close']), 0)
    df.drop(df.index[0], inplace = True)
    df['avg_gain'] = pd.Series(dtype = object)
    df.iloc[n-1, -1] = df.iloc[:n, -3].mean()
    df['avg_loss'] = pd.Series(dtype = object)
    df.iloc[n-1, -1] = df.iloc[:n, -3].mean()
    for i in list(range(n,len(df))):
        df.iloc[i, -2] = ((df.iloc[i-1, -2]*13)+df.iloc[i, -4])/n
        df.iloc[i, -1] = ((df.iloc[i-1, -1]*13)+df.iloc[i, -3])/n
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    return df['RSI']

def cagr(df, n):
    df = df.copy()
    df["cum_return"] = (1 + df["daily_return"]).cumprod()
    cagr = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return cagr

tickers = ['ADANIPORTS.NS',
 'ASIANPAINT.NS',
 'AXISBANK.NS',
 'BAJAJ-AUTO.NS',
 'BAJFINANCE.NS',
 'BAJAJFINSV.NS',
 'BPCL.NS',
 'BHARTIARTL.NS',
 'INFRATEL.NS']

start = dt.date.today() - dt.timedelta(40+365*1)
# start = '2014-12-31'
end = dt.date.today()
# end = '2019-12-31'
ohlcv_dict = {}

for ticker in tickers:
    ohlcv_dict[ticker] = yf.download(ticker, start, end, interval = '1d')
    ohlcv_dict[ticker].dropna(inplace = True, how = 'all')
tickers = ohlcv_dict.keys()


ohlcv = copy.deepcopy(ohlcv_dict)
return_df = pd.DataFrame()
for ticker in tickers:
    print("calculating monthly return for ",ticker)
    ohlcv[ticker]["mon_ret"] = ohlcv[ticker]['Adj Close'].pct_change()
    return_df[ticker] = ohlcv[ticker]["mon_ret"]
return_df.drop(return_df.index[:2], inplace = True)


ohlcv_rsi = pd.DataFrame()
for ticker in tickers:
    ohlcv_rsi[ticker] = rsi(ohlcv[ticker])
    
trade = pd.DataFrame(index = return_df.index, columns = tickers)

for ticker in tickers:
    position = ''
    for i in list(range(len(ohlcv_rsi))):
        if ohlcv_rsi.loc[ohlcv_rsi.index[i],ticker] < 30 and position != 'buy':
            position = 'buy'
        elif ohlcv_rsi.loc[ohlcv_rsi.index[i],ticker] > 70 and position == 'buy':
            position = 'sell'
            trade.loc[trade.index[i], ticker] = return_df.loc[return_df.index[i], ticker]
        elif position == 'buy':
            trade.loc[trade.index[i], ticker] = return_df.loc[return_df.index[i], ticker]

trade['daily_return'] = trade.mean(axis = 1)
trade['daily_return'].fillna(0, inplace = True)
trade['worth'] = pd.Series(dtype = object)
trade.loc[trade.index[0],'worth'] = 100

for i in list(range(1, len(trade))):    
    trade.loc[trade.index[i], 'worth'] = trade.loc[trade.index[i-1],'worth']*(1+trade.loc[trade.index[i], 'daily_return'])

check = pd.DataFrame(index = return_df.index)
for ticker in tickers:
    check[ticker]= trade[ticker]
    check[ticker+' RSI'] = ohlcv_rsi[ticker]
    check[ticker+ ' d_ret'] = return_df[ticker]

cagr(trade, 4.8)
