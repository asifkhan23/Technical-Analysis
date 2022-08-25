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
ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
t = st.sidebar.selectbox('Select Number of Days', (400, 350, 180, 90, 60, 45, 30, 15, 10, 7, 5, 3, 2, 1))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1h', '15m', '1m'))
st.header(f'{ticker.upper()} Technical Indicators')


# In[2]:


start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)


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
    

fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['200 MA'], name='200 MA',
                         line = dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['50 MA'], name='50 SMA',
                         line = dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['9 MA'], name='9 SMA',
                         line = dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['21 MA'], name='21 SMA',
                         line = dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['100 MA'], name='100 SMA',
                         line = dict(color='purple', width=2)))



layout = go.Layout(
    title=f"{ticker.upper()} Moving Averages & Parabolic Stop & Reverse",
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=600, width=800)

if i == '1d':
    fig.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
else:
    fig.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )    


fig.update_layout(layout)
    
# fig.show()


# In[13]:


df['SMA'] = df['Adj Close'].rolling(20).mean()
df['Std Dev'] = df['Adj Close'].rolling(20).std()
df['Upper'] = df['SMA'] + 2 * df['Std Dev']
df['Lower'] = df['SMA'] - 2 * df['Std Dev']


# In[14]:


fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'])])

fig2.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upperband',
                         line = dict(color='Black', width=2)))

fig2.add_trace(go.Scatter(x=df.index, y=df['SMA'], name='Middleband',
                         line = dict(color='turquoise', width=2)))

fig2.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lowerband',
                         line = dict(color='Black', width=2)))


layout = go.Layout(
    title=f'{ticker.upper()} Bollinger Bands',
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=600, width=800,
    )

if i == '1d':
    fig2.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
else:
    fig2.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )

fig2.update_layout(layout)
    
# fig2.show()


# In[15]:


# Construct a 2 x 1 Plotly figure
fig3 = make_subplots(rows=6, cols=1, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "RSI", "MACD",  "ATR", 'ADX', 'Stochastic Oscillators'))

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

fig3.append_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                         line = dict(color='green', width=4)), row = 2, col = 1)

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


# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1600, width=1000,
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

# fig3.show()

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
max_level = df['High'][highest_swing]
min_level = df['Low'][lowest_swing]
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["PSAR & SMA", "Bollinger Bands", "Oscillators", "Fibonacci Retracements", 'Dow Theory'])

with tab1:
    st.header("PSAR & SMA")
    st.plotly_chart(fig)

with tab2:
    st.header("Bollinger Bands")
    st.plotly_chart(fig2)

with tab3:
    st.header("Oscillators")
    st.plotly_chart(fig3)
    
with tab4:
    st.header("Fibonacci Retracements")
    st.write(" Retracemrnt Levels - (0=black, 0.236=red, 0.382=green, 0.5=blue, 0.618=cyan, 0.786= magenta,1 = yellow )")
    st.write(y)
    st.plotly_chart(fig4)

with tab5:
    st.header("Dow Theory")
    st.pyplot(fig5)

# st.write(fig)
# st.write(fig2)
# st.write(fig3)

