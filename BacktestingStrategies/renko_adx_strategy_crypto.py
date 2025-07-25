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
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models import BarSet
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

API_KEY = "PK7D1QOR39TVCDYW12I7"
SECRET_KEY = "St8DcSx6cnE2lJldpFrTm9zFTtbpNlS6VHoW2co8"

client = CryptoHistoricalDataClient()

def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    return df["ATR"]

def ADX(DF, n=20):    # Function taken from TradingView
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = ATR(DF, n)
    df["upmove"] = df["high"] - df["high"].shift(1)
    df["downmove"] = df["low"].shift(1) - df["low"]
    df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean() #resultados mas precisos usando 'alpha' en vez de 'span' o 'com'
    df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()

    return df["ADX"]

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252*390)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252*390)  #intervalos de 5 mins
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
    

# def renko_DF(DF, hourly_df):
#     "function to convert ohlc data into renko bricks"
#     df = DF.copy()
#     df.reset_index(inplace=True)  # Cambia el 'Date' de indice a columna (requerido para la funcion Renko)
#     df.columns = ["date","open","high","low","close","volume"] # Renombrar columnas según libreria de Renko
#     df2 = Renko(df)  # Crear instancia de Renko
#     df2.brick_size = 3*round(ATR(hourly_df, 120).iloc[-1],0) # Define 'bricksize' como la ULTIMA OBSERVACION del ATR
#     renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
#     return renko_df


# df = pd.read_csv('D:\\quant\\data\\sp500_july2025.csv')
# tickers = np.array(df["Symbol"]).tolist()

ohlc_intraday = {} # directory with ohlc value for each stock  

start = dt.datetime.today()-dt.timedelta(360)
end = dt.datetime.today()



# Filtrar solo los activos que están activos y tienen par USD
tickers = symbols = ["AAVE/USD", "ADA/USD", "ALGO/USD", "APE/USD", "AVAX/USD", "BAT/USD", 
                     "BCH/USD", "BTC/USD", "COMP/USD", "CRV/USD", "DOGE/USD", "DOT/USD", 
                     "EOS/USD", "ETC/USD", "ETH/USD", "FIL/USD", "GRT/USD", "LINK/USD", 
                     "LTC/USD", "MANA/USD", "MATIC/USD", "MKR/USD", "SHIB/USD", "SNX/USD", 
                     "SOL/USD", "SUSHI/USD", "UNI/USD", "USDT/USD", "XLM/USD", "XMR/USD", 
                     "XRP/USD", "XTZ/USD", "YFI/USD", "ZEC/USD"]



# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    try:
        req = CryptoBarsRequest(
            symbol_or_symbols= [ticker],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )
        
        # Obtener datos
        bars = client.get_crypto_bars(req)

        # Convertir a DataFrame
        df = bars.df
        #df = df.droplevel(0)
        df = df.reset_index()
      

        if df.empty:
            print(f"{ticker} no tiene datos.")
            continue
        df.dropna(how="all", inplace=True)
        ohlc_intraday[ticker] = df
    except Exception as e:
        print(f"Error con {ticker}: {e}")
        continue
    
    
################################Backtesting####################################

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    # df.reset_index(inplace=True) --- No es necesario pues ya se hizo el reset_index al procesar los datos
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120).iloc[-1],0))
    renko_df = df2.period_close_bricks()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    
    for i in range(1,len(renko_df["bar_num"])):  # Hace 'suma acumulativa' de las filas de bar_num
        
        if renko_df.loc[i, "bar_num"] > 0 and renko_df.loc[i-1, "bar_num"] > 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]
            
        elif renko_df.loc[i, "bar_num"] < 0 and renko_df.loc[i-1, "bar_num"] < 0:
            renko_df.loc[i, "bar_num"] += renko_df.loc[i-1, "bar_num"]
            
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True) # Quita fechas repetidas
    
    return renko_df
    

#Merging renko df with original ohlc df
ohlc_renko = {}
df = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}

for ticker in tickers:
    print("merging for ",ticker)
    renko = renko_DF(df[ticker]) # DataFrame 'renko' con columna "bar_num"
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    df_reset = df[ticker].copy()
    df_reset.rename(columns={"timestamp": "Date"}, inplace=True)  # renombra para que coincida con renko
    
    renko["Date"] = pd.to_datetime(renko["Date"]).dt.tz_localize(None)
    df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.tz_localize(None)

    ohlc_renko[ticker] = df_reset.merge(   # Agrega al DataFrame original la columan "bar_num" de Renko y hace el Merge con "Date"
        renko[["Date", "bar_num"]],
        how="outer",
        on="Date"
                                        )
    ohlc_renko[ticker]["bar_num"] = ohlc_renko[ticker]["bar_num"].ffill()
    ohlc_renko[ticker]["ADX"]=  ADX(ohlc_renko[ticker])
    
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []
    
    

#Identifying signals and calculating daily return
for ticker in tickers:
    print("calculating daily returns for ",ticker)
    
    for i in range(len(ohlc_intraday[ticker])):
        
        if tickers_signal[ticker] == "":
            
            tickers_ret[ticker].append(0)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i] >= 2 and ohlc_renko[ticker]["ADX"][i] >= 25:
                    tickers_signal[ticker] = "Buy"
                elif ohlc_renko[ticker]["bar_num"][i] <=-2 and ohlc_renko[ticker]["ADX"][i] >= 25:
                    tickers_signal[ticker] = "Sell"
        
        elif tickers_signal[ticker] == "Buy":
            
            tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i]/ohlc_renko[ticker]["close"][i-1])-1)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["ADX"][i] >= 25:
                    tickers_signal[ticker] = "Sell"
                elif ohlc_renko[ticker]["bar_num"][i] < 2 and ohlc_renko[ticker]["ADX"][i] < 25:
                    tickers_signal[ticker] = ""
                
        elif tickers_signal[ticker] == "Sell":
            tickers_ret[ticker].append((ohlc_renko[ticker]["close"][i-1]/ohlc_renko[ticker]["close"][i])-1)
            if i > 0:
                if ohlc_renko[ticker]["bar_num"][i] >= 2 and ohlc_renko[ticker]["ADX"][i] >= 25:
                    tickers_signal[ticker] = "Buy"
                elif ohlc_renko[ticker]["bar_num"][i] > -2 and ohlc_renko[ticker]["ADX"][i] < 25:
                    tickers_signal[ticker] = ""
                    
    ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])
    


#calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]["ret"]
strategy_df["ret"] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df,0.044)
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
    sharpe_ratios[ticker] =  sharpe(ohlc_renko[ticker],0.044)
    max_drawdown[ticker] =  max_dd(ohlc_renko[ticker])
    

KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
KPI_df.T
KPI_df["Overall"] = KPI_df.mean(axis=1)

print("---------------------------Strategy KPIs-----------------------------------------------")
print(CAGR(strategy_df))
print(sharpe(strategy_df,0.044))
print(max_dd(strategy_df))
print("------------------------------------------------------------------------------------------")
