#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import talib
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mplfinance as mpf
from plotly.subplots import make_subplots
import streamlit as st
t= st.sidebar.selectbox('Select one symbol', ( '400', '60',"15"))
ticker = st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT', 'TQQQ', 'TSLA', 'META', 'NFLX', 'GOOG', 'NVDA','AMD', 'QQQ', 'UPRO', 'SQQQ'))
t= st.sidebar.selectbox('Select one symbol', ( '1d', '1h',"15m"))
st.header(f'{ticker} Technical Analysis')


# In[6]:


def trading_algo(t, ticker, i):
    start = dt.datetime.today()-dt.timedelta(t)
    end = dt.datetime.today()
    df = yf.download(ticker, start, end, interval = i)
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Adj Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['slowk'] = slowk
    df['slowd'] = slowd
    df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
    df['ROCR'] = talib.ROCR(df['Adj Close'], timeperiod=10)
    macd, macdsignal, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    df['50 MA'], df['200 MA'], df['100 MA'] = talib.MA(df['Adj Close'], timeperiod=50, matype=0), talib.MA(df['Adj Close'], timeperiod=200, matype=0), talib.MA(df['Adj Close'], timeperiod=100, matype=0)
    df['9 MA'], df['21 MA'] = talib.MA(df['Adj Close'], timeperiod=9, matype=0), talib.MA(df['Adj Close'], timeperiod=21, matype=0)
    df['PSAR'] = real = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df['Adj Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
    df.dropna(inplace=True)
    return df


# In[7]:


df = trading_algo(400, ticker, '1d')
df.tail(10)


# In[33]:


fig1 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'])])

fig1.add_trace(go.Scatter(x=df.index, y=df['PSAR'], name='PSAR', mode = 'markers',
                         marker = dict(color='black', size=4)))

fig1.add_trace(go.Scatter(x=df.index, y=df['200 MA'], name='200 SMA',
                         line = dict(color='red', width=2)))

fig1.add_trace(go.Scatter(x=df.index, y=df['50 MA'], name='50 SMA',
                         line = dict(color='green', width=2)))

fig1.add_trace(go.Scatter(x=df.index, y=df['9 MA'], name='9 SMA',
                         line = dict(color='blue', width=2)))

fig1.add_trace(go.Scatter(x=df.index, y=df['21 MA'], name='21 SMA',
                         line = dict(color='orange', width=2)))

fig1.add_trace(go.Scatter(x=df.index, y=df['100 MA'], name='100 SMA',
                         line = dict(color='purple', width=2)))

layout = go.Layout(
    title=f"{ticker} Moving Averages & Parabolic Stop & Reverse",
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=500, width=900,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    ))

fig1.update_layout(layout)
    
# fig1.show()


# In[35]:


fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'])])

fig2.add_trace(go.Scatter(x=df.index, y=df['upperband'], name='Upperband',
                         line = dict(color='Black', width=2)))

fig2.add_trace(go.Scatter(x=df.index, y=df['middleband'], name='Middleband',
                         line = dict(color='turquoise', width=2)))

fig2.add_trace(go.Scatter(x=df.index, y=df['lowerband'], name='Lowerband',
                         line = dict(color='Black', width=2)))


layout = go.Layout(
    title=f'{ticker} Bollinger Bands',
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=500, width=900,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    ))

fig2.update_layout(layout)
    
# fig2.show()


# In[41]:


# Construct a 2 x 1 Plotly figure
fig3 = make_subplots(rows=6, cols=1, subplot_titles=(f"{ticker} Daily Candlestick Chart", "RSI", "MACD",  "ATR", 'ADX', 'Stochastic Oscillators'))

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

fig3.append_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                         line = dict(color='green', width=4)), row = 2, col = 1)

# Fast Signal (%k)
fig3.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macd'],
        line=dict(color='red', width=2),
        name='macd',
        # showlegend=False,
        legendgroup='2',
    ), row=3, col=1
)
# Slow signal (%d)
fig3.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macdsignal'],
        line=dict(color='red', width=2),
        # showlegend=False,
        legendgroup='2',
        name='signal'
    ), row=3, col=1
)
# Colorize the histogram values
colors = np.where(df['macdhist'] < 0, 'red', 'green')
# Plot the histogram
fig3.append_trace(
    go.Bar(
        x=df.index,
        y=df['macdhist'],
        name='histogram',
        marker_color=colors,
    ), row=3, col=1
)


fig3.append_trace(go.Scatter(x=df.index, y=df['ATR'], name='Average True Range',
                         line = dict(color='royalblue', width=4)), row = 4, col = 1 )

fig3.append_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX',
                         line = dict(color='red', width=4)), row = 5, col = 1)

fig3.append_trace(go.Scatter(x=df.index, y=df['slowk'], name='Fast K',
                         line = dict(color='blue', width=2)), row = 6, col = 1)

fig3.append_trace(go.Scatter(x=df.index, y=df['slowd'], name='Slow D',
                         line = dict(color='red', width=2)), row = 6, col = 1)


# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=2000, width=1400,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)

# Update options and show plot
fig3.update_layout(layout)

# fig3.show()


# In[42]:


st.write(fig1)
st.write(fig2)
st.write(fig3)


# In[ ]:




