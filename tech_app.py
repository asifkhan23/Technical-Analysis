#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.lines import Line2D
from collections import deque
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import statistics as stat
from scipy.stats import linregress
import math
ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical Indicators')


# In[2]:


start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)
# df = yf.download(ticker, period=t, interval= i)


# In[3]:


def ATR(df,n):
    "function to calculate True Range and Average True Range"
    df = df.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    dfx = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return dfx


# In[4]:


df['Fast_EMA']=df['Adj Close'].ewm(span = 12, min_periods = 12).mean()
df['Slow_EMA']=df['Adj Close'].ewm(span = 26, min_periods = 26).mean()
df['MACD'] = df['Fast_EMA']-df['Slow_EMA']
df['Signal'] = df['MACD'].ewm(span = 9, min_periods = 9).mean()
df['Histogram'] = df['MACD'] - df['Signal']


# In[5]:


df['H-L']=abs(df['High']-df['Low'])
df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
df['ATR'] = df['TR'].rolling(14).mean()


# In[6]:


df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)
df["gain"] = np.where(df["change"]>=0, df["change"], 0)
df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
df["avgGain"] = df["gain"].ewm(alpha=1/14, min_periods=14).mean()
df["avgLoss"] = df["loss"].ewm(alpha=1/14, min_periods=14).mean()
df["rs"] = df["avgGain"]/df["avgLoss"]
df["rsi"] = 100 - (100/ (1 + df["rs"]))


# In[7]:


dfz = ATR(df, 14)
df["upmove"] = df["High"] - df["High"].shift(1)
df["downmove"] = df["Low"].shift(1) - df["Low"]
df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
df["+di"] = 100 * (df["+dm"]/dfz["ATR"]).ewm(alpha=1/14, min_periods=14).mean()
df["-di"] = 100 * (df["-dm"]/dfz["ATR"]).ewm(alpha=1/14, min_periods=14).mean()
df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/14, min_periods=14).mean()


# In[8]:


high_roll = df["High"].rolling(14).max()
low_roll = df["Low"].rolling(14).min()

# Fast stochastic indicator
num = df["Adj Close"] - low_roll
denom = high_roll - low_roll
df["K"] = (num / denom) * 100

# Slow stochastic indicator
df["D"] = df["K"].rolling(3).mean()


# Supertrend

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)
    
    
atr_period = 10
atr_multiplier = 3.0


supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)
# In[9]:


def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = list(df.index)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    psar = close[0:len(close)]
    psarbull = [None] * length # Bullish signal - dot below candle
    psarbear = [None] * length # Bearish signal - dot above candle
    bull = True
    af = iaf # acceleration factor
    ep = low[0] # ep = Extreme Point
    hp = high[0] # High Point
    lp = low[0] # Low Point

    # https://www.investopedia.com/terms/p/parabolicindicator.asp - Parabolic Stop & Reverse Formula from Investopedia 
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}


# In[12]:


if __name__ == "__main__":
    import sys
    import os
    
    startidx = 0
    endidx = len(df)
    
    result = psar(df)
    dates = result['dates'][startidx:endidx]
    close = result['close'][startidx:endidx]
    psarbear = result['psarbear'][startidx:endidx]
    psarbull = result['psarbull'][startidx:endidx]
    df['200 MA'] = df['Adj Close'].rolling(200).mean()
    df['100 MA'] = df['Adj Close'].rolling(100).mean()
    df['50 MA'] = df['Adj Close'].rolling(50).mean()
    df['21 MA'] = df['Adj Close'].rolling(21).mean()
    df['9 MA'] = df['Adj Close'].rolling(9).mean()
    

# fig = go.Figure(data=[go.Candlestick(x=df.index,
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Close'])])

# fig.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
#                          marker = dict(color='green', size=2)))

# fig.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
#                          marker = dict(color='red', size=2)))

# fig.add_trace(go.Scatter(x=df.index, y=df['200 MA'], name='200 MA',
#                          line = dict(color='red', width=2)))

# fig.add_trace(go.Scatter(x=df.index, y=df['50 MA'], name='50 SMA',
#                          line = dict(color='green', width=2)))

# fig.add_trace(go.Scatter(x=df.index, y=df['9 MA'], name='9 SMA',
#                          line = dict(color='blue', width=2)))

# fig.add_trace(go.Scatter(x=df.index, y=df['21 MA'], name='21 SMA',
#                          line = dict(color='orange', width=2)))

# fig.add_trace(go.Scatter(x=df.index, y=df['100 MA'], name='100 SMA',
#                          line = dict(color='purple', width=2)))



# layout = go.Layout(
#     title=f"{ticker.upper()} Moving Averages & Parabolic Stop & Reverse",
#     plot_bgcolor='#efefef',
#     # Font Families
#     font_family='Monospace',
#     font_color='#000000',
#     font_size=15,
#     height=600, width=800)

# if i == '1d':
#     fig.update_xaxes(
#             rangeslider_visible=True,
#             rangebreaks=[
#                 # NOTE: Below values are bound (not single values), ie. hide x to y
#                 dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                 # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                     # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                 ]
#                     )
# else:
#     fig.update_xaxes(
#             rangeslider_visible=True,
#             rangebreaks=[
#                 # NOTE: Below values are bound (not single values), ie. hide x to y
#                 dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                 dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                     # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                 ]
#                     )    


# fig.update_layout(layout)
    
# fig.show()


# In[13]:


df['SMA'] = df['Adj Close'].rolling(20).mean()
df['Std Dev'] = df['Adj Close'].rolling(20).std()
df['Upper'] = df['SMA'] + 2 * df['Std Dev']
df['Lower'] = df['SMA'] - 2 * df['Std Dev']

# Regression Channels

nomalized_return=np.log(df['Adj Close']/df['Adj Close'].iloc[0])

dfr = pd.DataFrame(data=nomalized_return)

# dfr = dfr.resample('D').asfreq()

# Create a 'x' and 'y' column for convenience
dfr['y'] = dfr['Adj Close']     # create a new y-col (optional)
dfr['x'] = np.arange(len(dfr))  # create x-col of continuous integers


# Drop the rows that contain missing days
dfr = dfr.dropna()

X=dfr['x'].values[:, np.newaxis]
y=dfr['y'].values[:, np.newaxis]

# Fit linear regression model using scikit-learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Make predictions w.r.t. 'x' and store it in a column called 'y_pred'
dfr['y_pred'] = lin_reg.predict(dfr['x'].values[:, np.newaxis])

dfr['above']= y + np.std(y)
dfr['below']= y - np.std(y)
# Plot 'y' and 'y_pred' vs 'DateTimeIndex`

# df['above_us'] = lin_reg.predict(df['above'].values[:, np.newaxis])
# df['below_us'] = lin_reg.predict(df['below'].values[:, np.newaxis])

# df[['y', 'y_pred', 'above_us', 'below_us']].plot(figsize = (12,6), title = 'Regression Analysis')

dfr['y_unscaled'] = df['Adj Close']
dfr['y_pred_unscaled'] = np.exp(dfr['y_pred']) * df['Adj Close'].iloc[0]

data_len = len(df)
df['Number'] = np.arange(data_len)+1
df_high = df.copy()
df_low = df.copy()

while len(df_high)>3:
    slope, intercept, r_value, p_value, std_err = linregress(x=df_high['Number'], y=df_high['High'])
    df_high = df_high.loc[df_high['High'] > slope * df_high['Number'] + intercept]

while len(df_low)>3:
    slope, intercept, r_value, p_value, std_err = linregress(x=df_low['Number'], y=df_low['Low'])
    df_low = df_low.loc[df_low['Low'] < slope * df_low['Number'] + intercept]

slope, intercept, r_value, p_value, std_err = linregress(x=df_high['Number'], y=df_high['Close'])
df['Uptrend'] = slope * df['Number'] + intercept

slope, intercept, r_value, p_value, std_err = linregress(x=df_low['Number'], y=df_low['Close'])
df['Downtrend'] = slope * df['Number'] + intercept

# Ichimoku Cloud

def ichimoku_cloud(df, a, b, c):
  df = df.copy()
  nine_h = df['High'].rolling(a).max()
  nine_l = df['Low'].rolling(a).min()
  tsix_h = df['High'].rolling(b).max()
  tsix_l = df['Low'].rolling(b).min()
  df['Fast_Moving_Average'] = (nine_h + nine_l)/2
  df['Slow_Moving_Average'] = (tsix_h + tsix_l)/2
  df['Leading_Span_A'] = ((df['Fast_Moving_Average'] + df['Slow_Moving_Average'])/2).shift(b)
  fifty2_h = df['High'].rolling(c).max()
  fifty2_l = df['Low'].rolling(c).min()
  df['Leading_Span_B'] = ((fifty2_h + fifty2_l)/2).shift(b)
  df['Chikou_Span'] = df['Adj Close'].shift(-b)
  df['Action'] = np.where(df['Fast_Moving_Average'] > df['Slow_Moving_Average'], 1, 0)
  df['Action'] = np.where(df['Fast_Moving_Average'] < df['Fast_Moving_Average'], -1, df['Action'])
  df['Entry'] = df.Action.diff()
  return df
df5 = ichimoku_cloud(df, 9, 26, 52)

candle = go.Candlestick(x=df5.index, open=df5['Open'],
                       high=df5['High'], low=df5['Low'],
                       close=df5['Close'], name='Candlestick')
# Sets fill color to green when value greater or equal to 1
# and red otherwise
def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'

df_c = df5.copy()

# Sine Waves

def super_smoother(data, length):
    """Python implementation of the Super Smoother indicator created by John Ehlers
    :param data: list of price data
    :type data: list
    :param length: period
    :type length: int
    :return: Super smoothed price data
    :rtype: list
    """
    ssf = []
    for i, _ in enumerate(data):
        if i < 2:
            ssf.append(0)
        else:
            arg = 1.414 * 3.14159 / length
            a_1 = math.exp(-arg)
            b_1 = 2 * a_1 * math.cos(4.44/float(length))
            c_2 = b_1
            c_3 = -a_1 * a_1
            c_1 = 1 - c_2 - c_3
            ssf.append(c_1 * (data[i] + data[i-1]) / 2 + c_2 * ssf[i-1] + c_3 * ssf[i-2])
    return ssf

def ebsw(data, hp_length, ssf_length):
    """Python implementation of Even Better Sine Wave indicator created by John Ehlers
    :param data: list of price data
    :type data: list
    :param hp_length: period
    :type hp_length: int
    :param ssf_length: predict
    :type ssf_length: int
    :return: Even Better Sine Wave indicator data
    :rtype: list
    """
    pi = 3.14159
    alpha1 = (1 - math.sin(2 * pi / hp_length)) / math.cos(2 * pi / hp_length)

    hpf = []

    for i, _ in enumerate(data):
        if i < hp_length:
            hpf.append(0)
        else:
            hpf.append((0.5 * (1 + alpha1) * (data[i] - data[i - 1])) + (alpha1 * hpf[i - 1]))

    ssf = super_smoother(hpf, ssf_length)

    wave = []
    for i, _ in enumerate(data):
        if i < ssf_length:
            wave.append(0)
        else:
            w = (ssf[i] + ssf[i - 1] + ssf[i - 2]) / 3
            p = (pow(ssf[i], 2) + pow(ssf[i - 1], 2) + pow(ssf[i - 2], 2)) / 3
            if p == 0:
                wave.append(0)
            else:
                wave.append(w / math.sqrt(p))

    return wave

df['ebs'] = ebsw(df['Adj Close'], 40, 10)
df['ebs_signal_buy'] = np.where(df["ebs"]> -0.50, 1, 0)
df['ebs_p'] = df['ebs_signal_buy'] * df['ebs']
df['ebs_signal_sell'] = np.where(df["ebs"]< 0.50, 1, 0)
df['ebs_n'] = df['ebs_signal_sell'] * df['ebs']
df['ebs_p'].replace(0.000000, np.nan, inplace=True)
df['ebs_n'].replace(0.000000, np.nan, inplace=True)

# Elher's Decycler

def decycler(data, hp_length):
    """Python implementation of Simple Decycler indicator created by John Ehlers
    :param data: list of price data
    :type data: list
    :param hp_length: High Pass filter length
    :type hp_length: int
    :return: Decycler applied price data
    :rtype: list
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])

    dec = []
    for i, _ in enumerate(data):
        dec.append(data[i] - hpf[i])

    return dec

df['decycler'] = decycler(df['Adj Close'], 20)
df['decycler_signal_buy'] = np.where(df["decycler"]<df['Adj Close'], 1, 0)
df['decycler_p'] = df['decycler_signal_buy'] * df['decycler']
df['decycler_signal_sell'] = np.where(df["decycler"]>df['Adj Close'], 1, 0)
df['decycler_n'] = df['decycler_signal_sell'] * df['decycler']
df['decycler_p'].replace(0.000000, np.nan, inplace=True)
df['decycler_n'].replace(0.000000, np.nan, inplace=True)
# In[14]:


# fig2 = go.Figure(data=[go.Candlestick(x=df.index,
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Adj Close'])])

# fig2.add_trace(go.Scatter(x=df.index, y=df['9 MA'], name='9 SMA',
#                          line = dict(color='blue', width=2)))

# fig2.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upperband',
#                          line = dict(color='Black', width=2)))

# fig2.add_trace(go.Scatter(x=df.index, y=df['SMA'], name='Middleband',
#                          line = dict(color='orange', width=2)))

# fig2.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lowerband',
#                          line = dict(color='Black', width=2)))

# fig2.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
#                          marker = dict(color='green', size=2)))

# fig2.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
#                          marker = dict(color='red', size=2)))

# fig2.add_trace(go.Scatter(x=df.index, y=df['200 MA'], name='200 MA',
#                          line = dict(color='red', width=2), visible='legendonly'))

# fig2.add_trace(go.Scatter(x=df.index, y=df['50 MA'], name='50 SMA',
#                          line = dict(color='green', width=2), visible='legendonly'))

# fig2.add_trace(go.Scatter(x=df.index, y=df['100 MA'], name='100 SMA',
#                          line = dict(color='purple', width=2), visible='legendonly'))

# fig2.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band',
#                          line = dict(color='green', width=2), visible='legendonly'))

# fig2.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band',
#                          line = dict(color='red', width=2), visible='legendonly'))

# layout = go.Layout(
#     title=f'{ticker.upper()} Bollinger Bands',
#     plot_bgcolor='#efefef',
#     # Font Families
#     font_family='Monospace',
#     font_color='#000000',
#     font_size=15,
#     height=600, width=800,
#     )

# if i == '1d':
#     fig2.update_xaxes(
#             rangeslider_visible=True,
#             rangebreaks=[
#                 # NOTE: Below values are bound (not single values), ie. hide x to y
#                 dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                 # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                     # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                 ]
#                     )
# else:
#     fig2.update_xaxes(
#             rangeslider_visible=True,
#             rangebreaks=[
#                 # NOTE: Below values are bound (not single values), ie. hide x to y
#                 dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                 dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                     # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                 ]
#                     )

# fig2.update_layout(layout)
    
# fig2.show()

# In[15]:


# Construct a 2 x 1 Plotly figure
fig3 = make_subplots(rows=7, cols=1, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "RSI", "MACD",  "ATR", 'ADX', 'Stochastic Oscillators'))

fig3.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Adj Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1
)


fig3.add_trace(go.Scatter(x=df.index, y=df['9 MA'], name='9 SMA',
                         line = dict(color='blue', width=2),visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upperband',
                         line = dict(color='Black', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['SMA'], name='20 SMA',
                         line = dict(color='orange', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lowerband',
                         line = dict(color='Black', width=2),visible='legendonly'))

fig3.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig3.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))

fig3.add_trace(go.Scatter(x=df.index, y=df['200 MA'], name='200 SMA',
                         line = dict(color='red', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['50 MA'], name='50 SMA',
                         line = dict(color='green', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['100 MA'], name='100 SMA',
                         line = dict(color='purple', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band',
                         line = dict(color='green', width=2)))

fig3.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band',
                         line = dict(color='red', width=2)))

fig3.add_trace(go.Scatter(x=df.index, y=dfr['y_pred_unscaled'], name='Regression',
                          line = dict(color='blue', width=2),visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['Uptrend'], name='Resistance',
                         line = dict(color='red', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['Downtrend'], name='Support',
                         line = dict(color='green', width=2),visible='legendonly'))

fig3.append_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                         line = dict(color='green', width=4)), row = 2, col = 1)

fig3.add_trace(go.Scatter(x=df.index, y=df['decycler_p'], name='Decycler Bull',
                         line = dict(color='green', width=2), visible='legendonly'))

fig3.add_trace(go.Scatter(x=df.index, y=df['decycler_n'], name='Decycler Bear',
                         line = dict(color='red', width=2), visible='legendonly'))

# Fast Signal (%k)
fig3.append_trace(
    go.Scatter(
        x=df.index,
        y=df['MACD'],
        line=dict(color='blue', width=2),
        name='macd',
        # showlegend=False,
        legendgroup='2',
    ), row=3, col=1
)
# Slow signal (%d)
fig3.append_trace(
    go.Scatter(
        x=df.index,
        y=df['Signal'],
        line=dict(color='red', width=2),
        # showlegend=False,
        legendgroup='2',
        name='signal'
    ), row=3, col=1
)
# Colorize the histogram values
colors = np.where(df['Histogram'] < 0, 'red', 'green')
# Plot the histogram
fig3.append_trace(
    go.Bar(
        x=df.index,
        y=df['Histogram'],
        name='histogram',
        marker_color=colors,
    ), row=3, col=1
)


fig3.append_trace(go.Scatter(x=df.index, y=df['ATR'], name='Average True Range',
                         line = dict(color='royalblue', width=4)), row = 4, col = 1 )

fig3.append_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX',
                         line = dict(color='red', width=4)), row = 5, col = 1)

fig3.append_trace(go.Scatter(x=df.index, y=df['K'], name='Fast K',
                         line = dict(color='blue', width=2)), row = 6, col = 1)

fig3.append_trace(go.Scatter(x=df.index, y=df['D'], name='Slow D',
                         line = dict(color='red', width=2)), row = 6, col = 1)

fig3.append_trace(go.Scatter(x=df.index, y=df['ebs_p'], name='Sinewave Bull',
                         line = dict(color='green', width=2)), row = 7, col = 1 )

fig3.append_trace(go.Scatter(x=df.index, y=df['ebs_n'], name='Sinewave Bear',
                         line = dict(color='red', width=2)), row = 7, col = 1 )

# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=2800, width=1400,
)

if i == '1d':
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
elif i == '1wk':
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
else:
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )


# Update options and show plot
fig3.update_layout(layout)

fig3.show()

# Regression Channels Plot
# df_reg = yf.download(ticker, start, end, interval='1d')

fig6 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'])])


fig6.add_trace(go.Scatter(x=df.index, y=dfr['y_pred_unscaled'], name='Regression',
                          line = dict(color='blue', width=2)))

fig6.add_trace(go.Scatter(x=df.index, y=df['Uptrend'], name='Resistance',
                         line = dict(color='red', width=2)))

fig6.add_trace(go.Scatter(x=df.index, y=df['Downtrend'], name='Support',
                         line = dict(color='green', width=2)))


layout = go.Layout(
    title=f'{ticker.upper()} Regression Channels',
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=600, width=800,
    )

if i == '1d':
    fig6.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                #dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )    
else:
    fig6.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )

fig6.update_layout(layout)



# Where SpanA is greater than SpanB give label a value of 1 or 0 if not
df5['label'] = np.where(df5['Leading_Span_A'] > df5['Leading_Span_B'], 1, 0)

# Shift 1 period, compare dataframe for inequality with the cumulative 
# sum and store in group
df5['group'] = df5['label'].ne(df5['label'].shift()).cumsum()

# Get a groupby object that contains information on the group
df5 = df5.groupby('group')

# Cycle through the data pertaining to the fill between spans
dfs = []
for name, data in df5:
    dfs.append(data)

    
# Add 2 traces to the fig object for each time the spans cross
# and then define the fill using fill='tonexty' for the second trace
for df in dfs:
    fig3.add_traces(go.Scatter(x=df.index, y = df['Leading_Span_A'],
                              line = dict(color='rgba(0,0,0,0)'),visible='legendonly'))
    
    fig3.add_traces(go.Scatter(x=df.index, y = df['Leading_Span_B'],
                              line = dict(color='rgba(0,0,0,0)'),visible='legendonly',
                              fill='tonexty', 
                              fillcolor = get_fill_color(df['label'].iloc[0])))


# Create plots for all of the nonfill data
baseline = go.Scatter(x=df_c.index, y=df_c['Slow_Moving_Average'], 
                   line=dict(color='orange', width=2),visible='legendonly', name="Baseline")

conversion = go.Scatter(x=df_c.index, y=df_c['Fast_Moving_Average'], 
                  line=dict(color='blue', width=1),visible='legendonly', name="Conversionline")

lagging = go.Scatter(x=df_c.index, y=df_c['Chikou_Span'], 
                  line=dict(color='purple', width=2, dash='solid'),visible='legendonly', name="Lagging")

span_a = go.Scatter(x=df_c.index, y=df_c['Leading_Span_A'],
                  line=dict(color='green', width=2, dash='solid'),visible='legendonly', name="Span A")

span_b = go.Scatter(x=df_c.index, y=df_c['Leading_Span_B'],
                    line=dict(color='red', width=1, dash='solid'),visible='legendonly', name="Span B")

# Add plots to the figure
# fig7.add_trace(candle)
fig3.add_trace(baseline)
fig3.add_trace(conversion)
fig3.add_trace(lagging)
fig3.add_trace(span_a)
fig3.add_trace(span_b)



# Fibonacci

df1 = yf.download(ticker, start, end, interval = '1d')
df1.reset_index(inplace = True)

# highest_swing and lowest_swings generate the area for which we have to check ratios
highest_swing = -1
lowest_swing = -1
for i in range(1,df1.shape[0]-1):
  if df1['High'][i] > df1['High'][i-1] and df1['High'][i] > df1['High'][i+1] and (highest_swing == -1 or df1['High'][i] > df1['High'][highest_swing]):
    highest_swing = i
  if df1['Low'][i] < df1['Low'][i-1] and df1['Low'][i] < df1['Low'][i+1] and (lowest_swing == -1 or df1['Low'][i] < df1['Low'][lowest_swing]):
    lowest_swing = i

name = '0,0.236, 0.382, 0.5 , 0.618, 0.786,1'
ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
colors = ["black","red","green","blue","cyan","magenta","yellow"]
levels = []
max_level = df1['High'][highest_swing]
min_level = df1['Low'][lowest_swing]
for ratio in ratios:
  if highest_swing > lowest_swing: # Uptrend
    levels.append(max_level - (max_level-min_level)*ratio)
  else: # Downtrend
    levels.append(min_level + (max_level-min_level)*ratio)
    
# Fibonacci plot
fig4 = go.Figure()
fig4.add_traces(go.Candlestick(x=df1['Date'],
                              open=df1['Open'],
                              high=df1['High'],
                              low=df1['Low'],
                              close=df1['Close']))

start_date = df1['Date'][df1.index[min(highest_swing,lowest_swing)]]
end_date = df1['Date'][df1.index[max(highest_swing,lowest_swing)]]
y=np.array([start_date,end_date])
print(y)
for i in range(len(levels)):
  # previous pyplot plot
  # plt.hlines(levels[i],start_date, end_date,label="{:.1f}%".format(ratios[i]*100),colors=colors[i], linestyles="dashed")
  fig4.add_shape(type='line', 
    x0=start_date, y0=levels[i], x1=end_date, y1=levels[i],
    line=dict(
        color=colors[i],
        dash="dash"
    ))


# Dow Thoery

data = yf.download(ticker, start, end, interval = '1d')

data['local_max'] = data['Close'][
  (data['Close'].shift(1) < data['Close']) &
  (data['Close'].shift(-1) < data['Close'])]

data['local_min'] = data['Close'][
  (data['Close'].shift(1) > data['Close']) &
  (data['Close'].shift(-1) > data['Close'])]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# plt.figure(figsize=(15, 8))
# plt.plot(data['Close'], zorder=0)
# plt.scatter(data.index, data['local_max'], s=100,
#   label='Maxima', marker='^', c=colors[2])
# plt.scatter(data.index, data['local_min'], s=100,
#   label='Minima', marker='v', c=colors[1])
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.title(f'Local Maxima and Minima for {ticker}')
# plt.legend()
# plt.show()

max_idx = argrelextrema(data['Close'].values, np.greater, order=5)[0]
min_idx = argrelextrema(data['Close'].values, np.less, order=5)[0]
# plt.figure(figsize=(15, 8))
# plt.plot(data['Close'], zorder=0)
# plt.scatter(data.iloc[max_idx].index, data.iloc[max_idx]['Close'],
#   label='Maxima', s=100, color=colors[2], marker='^')
# plt.scatter(data.iloc[min_idx].index, data.iloc[min_idx]['Close'],
#   label='Minima', s=100, color=colors[1], marker='v')

# plt.legend()
# plt.show()

# Get K consecutive higher peaks
K = 2
high_idx = argrelextrema(data['Close'].values, np.greater, order=5)[0]
highs = data.iloc[high_idx]['Close']

extrema = []
ex_deque = deque(maxlen=K)
for i, idx in enumerate(high_idx):
  if i == 0:
    ex_deque.append(idx)
    continue
  if highs[i] < highs[i-1]:
    ex_deque.clear()

  ex_deque.append(idx)
  if len(ex_deque) == K:
    # K-consecutive higher highs found
    extrema.append(ex_deque.copy())

close = data['Adj Close'].values
dates = data.index

# plt.figure(figsize=(15, 8))
# plt.plot(data['Adj Close'])
# _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in extrema]
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.title(f'Higher Highs for {ticker} Closing Price')
# plt.legend(['Close', 'Consecutive Highs'])
# plt.show()

def getHigherLows(data: np.array, order=5, K=2):
  '''
  Finds consecutive higher lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be higher.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are higher than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] < lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerHighs(data: np.array, order=5, K=2):
  '''
  Finds consecutive lower highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be lower.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are lower than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] > highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getHigherHighs(data: np.array, order=5, K=2):
  '''
  Finds consecutive higher highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be higher.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=5)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] < highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerLows(data: np.array, order=5, K=2):
  '''
  Finds consecutive lower lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be lower.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are lower than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] > lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

from matplotlib.lines import Line2D

# close = data['Close'].values
# dates = data.index

# order = 5
# K = 2

# hh = getHigherHighs(close, order, K)
# hl = getHigherLows(close, order, K)
# ll = getLowerLows(close, order, K)
# lh = getLowerHighs(close, order, K)

# plt.figure(figsize=(15, 8))
# plt.plot(data['Close'])
# _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in hh]
# _ = [plt.plot(dates[i], close[i], c=colors[2]) for i in hl]
# _ = [plt.plot(dates[i], close[i], c=colors[3]) for i in ll]
# _ = [plt.plot(dates[i], close[i], c=colors[4]) for i in lh]
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.title(f'Potential Divergence Points for {ticker} Closing Price')
# legend_elements = [
#   Line2D([0], [0], color=colors[0], label='Close'),
#   Line2D([0], [0], color=colors[1], label='Higher Highs'),
#   Line2D([0], [0], color=colors[2], label='Higher Lows'),
#   Line2D([0], [0], color=colors[3], label='Lower Lows'),
#   Line2D([0], [0], color=colors[4], label='Lower Highs')
# ]
# plt.legend(handles=legend_elements)
# plt.show()

close = data['Close'].values
dates = data.index

order = 5
K = 2

hh = getHigherHighs(close, order, K)
hl = getHigherLows(close, order, K)
ll = getLowerLows(close, order, K)
lh = getLowerHighs(close, order, K)

fig5 = plt.figure(figsize=(15, 8))
plt.plot(data['Close'])
_ = [plt.plot(dates[i], close[i], c=colors[2]) for i in hh]
_ = [plt.plot(dates[i], close[i], c=colors[1]) for i in hl]
_ = [plt.plot(dates[i], close[i], c=colors[3]) for i in ll]
_ = [plt.plot(dates[i], close[i], c=colors[4]) for i in lh]

_ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
    c=colors[2], marker='^', s=100) for i in hh]
_ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
    c=colors[1], marker='^', s=100) for i in hl]
_ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
    c=colors[3], marker='v', s=100) for i in ll]
_ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]],
    c=colors[4], marker='v', s=100) for i in lh]
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title(f'Potential Divergence Points for {ticker} Closing Price')
legend_elements = [
  Line2D([0], [0], color=colors[0], label='Close'),
  Line2D([0], [0], color=colors[2], label='Higher Highs'),
  Line2D([0], [0], color='w',  marker='^',
         markersize=10,
         markerfacecolor=colors[2],
         label='Higher High Confirmation'),
  Line2D([0], [0], color=colors[1], label='Higher Lows'),
  Line2D([0], [0], color='w',  marker='^',
         markersize=10,
         markerfacecolor=colors[1],
         label='Higher Lows Confirmation'),
  Line2D([0], [0], color=colors[3], label='Lower Lows'),
  Line2D([0], [0], color='w',  marker='v',
         markersize=10,
         markerfacecolor=colors[3],
         label='Lower Lows Confirmation'),
  Line2D([0], [0], color=colors[4], label='Lower Highs'),
  Line2D([0], [0], color='w',  marker='v',
         markersize=10,
         markerfacecolor=colors[4],
         label='Lower Highs Confirmation')
]
plt.legend(handles=legend_elements)


tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Fibonacci Retracements", 'Dow Theory'])

with tab1:
    st.header("Technical Analysis")
    st.plotly_chart(fig3)
    
with tab2:
    st.header("Fibonacci")
    st.write(" Retracemrnt Levels - (0=black, 0.236=red, 0.382=green, 0.5=blue, 0.618=cyan, 0.786= magenta,1 = yellow )")
    st.write(y)
    st.plotly_chart(fig4)

with tab3:
    st.header("Dow Theory")
    st.pyplot(fig5)
