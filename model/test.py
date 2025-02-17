import yfinance as yf


df_yahoo = yf.download('AAPL', 
                       start='2000-01-01', 
                       end='2010-12-31',
                       progress=False)

print(df_yahoo)