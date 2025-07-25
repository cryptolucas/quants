# =============================================================================
# Import OHLCV data and calculate ATR technical indicator
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import yfinance as yf

# Download historical data for required stocks
tickers = ["AMZN","GOOG","MSFT"]
ohlcv_data = {}

# looping over tickers and storing OHLCV dataframe in dictionary
for ticker in tickers:
    temp = yf.download(ticker,period='1mo',interval='5m')
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker] = temp

def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False) #True Ranges es el #max de los 3 calculos previos
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()#se usa 'com' en vez de #'span' para calcular EMA ya que se obtiene más precision en este caso
    
    return df["ATR"]


for ticker in ohlcv_data:
    ohlcv_data[ticker]["ATR"] = ATR(ohlcv_data[ticker])
#Agregar la columna "ATR" al DataFrame original