# =============================================================================
# Backtesting strategy - IV : combining renko with other MACD
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

import numpy as np
import pandas as pd
from stocktrends import Renko
import statsmodels.api as sm
from alpha_vantage.timeseries import TimeSeries
import copy
import yfinance as yf
import datetime as dt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = "PK7D1QOR39TVCDYW12I7"
SECRET_KEY = "St8DcSx6cnE2lJldpFrTm9zFTtbpNlS6VHoW2co8"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['high']-df['low'])
    df['H-PC']=abs(df['high']-df['close'].shift(1))
    df['L-PC']=abs(df['low']-df['close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params.iloc[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)



def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    #df.reset_index(inplace=True)
    #df = df.iloc[:,[0,1,2,3,4,5]]
    #df.columns = ["date","open","high","low","close","volume"]
    df = df.copy()
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"].iloc[-1],0))
    renko_df = df2.period_close_bricks()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):  # Hace 'suma acumulativa' de las filas de bar_num
        if renko_df.loc[i, "bar_num"]>0 and renko_df.loc[i-1, "bar_num"]>0:
            renko_df.loc[i, "bar_num"]+=renko_df.loc[i-1, "bar_num"]
            
        elif renko_df.loc[i, "bar_num"]<0 and renko_df.loc[i-1, "bar_num"]<0:
            renko_df.loc[i, "bar_num"]+=renko_df.loc[i-1, "bar_num"]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True) # Quita fechas repetidas
    return renko_df




def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252*78)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252*78)  #intervalos de 5 mins
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


df = pd.read_csv('D:\\quant\\data\\sp500_july2025.csv')
tickers = np.array(df["Symbol"]).tolist()

ohlc_intraday = {} # directory with ohlc value for each stock  

start = dt.datetime.today()-dt.timedelta(360)
end = dt.datetime.today()


# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    try:
        request_params = StockBarsRequest(
                        symbol_or_symbols= [ticker],
                        timeframe=TimeFrame.Minute,  # por defecto es 1 minuto
                        start=start,
                        end=end,
                        )
        
        # Obtener datos
        bars = client.get_stock_bars(request_params)

        # Convertir a DataFrame
        df = bars.df
        df = df.droplevel(0)
        df = df.reset_index()
        print(df)

        if df.empty:
            print(f"{ticker} no tiene datos.")
            continue
        df.dropna(how="all", inplace=True)
        ohlc_intraday[ticker] = df
    except Exception as e:
        print(f"Error con {ticker}: {e}")
        continue

 
tickers = ohlc_intraday.keys() # redefine tickers variable after removing any tickers with corrupted data


################################Backtesting####################################

#Merging renko df with original ohlc df
ohlc_renko = {}
df = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}
for ticker in tickers:
    print("merging for ",ticker)
    renko = renko_DF(df[ticker])
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    df_reset = df[ticker].copy()
    #df_reset = df[ticker].reset_index()
    #df_reset.columns = df_reset.columns.get_level_values(0)
    df_reset.rename(columns={"timestamp": "Date"}, inplace=True)  # renombra para que coincida con renko
    
    renko["Date"] = pd.to_datetime(renko["Date"]).dt.tz_localize(None)
    df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.tz_localize(None)

    ohlc_renko[ticker] = df_reset.merge(
        renko[["Date", "bar_num"]],
        how="outer",
        on="Date"
                                        )
    ohlc_renko[ticker]["bar_num"] = ohlc_renko[ticker]["bar_num"].ffill()
    ohlc_renko[ticker]["macd"]= MACD(ohlc_renko[ticker],12,26,9)[0]
    ohlc_renko[ticker]["macd_sig"]= MACD(ohlc_renko[ticker],12,26,9)[1]
    ohlc_renko[ticker]["macd_slope"] = slope(ohlc_renko[ticker]["macd"],5)
    ohlc_renko[ticker]["macd_sig_slope"] = slope(ohlc_renko[ticker]["macd_sig"],5)
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []

       
#Identifying signals and calculating daily return
for ticker in tickers:
    print("calculating daily returns for ",ticker)
    for i in range(len(ohlc_intraday[ticker])):
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = "Buy"
                elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = "Sell"
        
        elif tickers_signal[ticker] == "Buy":
            tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i]/ohlc_renko[ticker]["close"][i-1])-1)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = "Sell"
                elif ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = ""
                
        elif tickers_signal[ticker] == "Sell":
            tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i-1]/ohlc_renko[ticker]["close"][i])-1)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = "Buy"
                elif ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                    tickers_signal[ticker] = ""
    ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])

#calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df,0.025)
max_dd(strategy_df)  

#visualizing strategy returns
(1+strategy_df["ret"]).cumprod().plot()

#calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
max_drawdown = {}
for ticker in tickers:
    print("calculating KPIs for ",ticker)      
    cagr[ticker] =  CAGR(ohlc_renko[ticker])
    sharpe_ratios[ticker] =  sharpe(ohlc_renko[ticker],0.025)
    max_drawdown[ticker] =  max_dd(ohlc_renko[ticker])
    

KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
KPI_df.T
KPI_df["Overall"] = KPI_df.mean(axis=1)
print(KPI_df)