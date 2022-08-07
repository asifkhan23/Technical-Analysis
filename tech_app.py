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
t = 400
ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
st.header(f'{ticker} Technical Indicators')


# In[2]:


start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end)


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
    title=f"{ticker} Moving Averages & Parabolic Stop & Reverse",
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=600, width=1400,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    ))

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
    title=f'{ticker} Bollinger Bands',
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=15,
    height=600, width=1400,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    ))

fig2.update_layout(layout)
    
# fig2.show()


# In[15]:


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


# In[16]:


st.write(fig)
st.write(fig2)
st.write(fig3)

