# =============================================================================
# Measuring the return of a buy and hold strategy - CAGR
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
    temp = yf.download(ticker,period='7mo',interval='1d')
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker] = temp

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["return"] = DF["Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()  # Se hace con producto acumulado en esta opcion
    n = len(df)/252   # 252 = Number of trading days
    CAGR = (df["cum_return"].iloc[-1])**(1/n) - 1
    return CAGR

# La otra opcion es usar la ecuacion directamente (más sencillo): [End value/Initial value] ^ (1/n) - 1 y DA LO MISMO...

for ticker in ohlcv_data:
    print("CAGR of {} = {}".format(ticker,CAGR(ohlcv_data[ticker])))