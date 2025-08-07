import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def getData(sT):
    stock = yf.Ticker(sT)
    getDataFromTime = stock.history(period='1y')
    getDataFromTime.reset_index(inplace=True)
    return getDataFromTime

#We are writing this part to basically allow for the model to find patterns
#to accurately predict stock price using real-time updated stock data.
def addFeatures(df):
    #Contains previous day's closing price (helps predict stock's short-term trends)
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)
    #Add column for 5-day & 10-day moving average.
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    #Measures stock's volatility
    df['STD_5'] = df['Close'].rolling(window=5).std()
    #Momentum & price movement analysis
    df['Daily_Return'] = df['Close'].pct_change()
    #Captures weekday effects & monthly trends
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df.dropna(inplace=True)
    return df

print("Hi! Welcome to my Stock Price Prediction program.")
print("Please enter a stock ticker below: ")
stock_ticker = input().upper()
dataFrameInfo = getData(stock_ticker)

if dataFrameInfo.empty:
    print("Couldn't fetch data for "+stock_ticker+". Please check that you've entered the right ticker.")
    exit()
else:
    print("This is a preview of your chosen stock's data: ")
    print(dataFrameInfo.head())
    dataFrameInfo = addFeatures(dataFrameInfo)
    
    x = dataFrameInfo[['Close_lag1', 'Close_lag2', 'Close_lag3', 'MA_5', 'MA_10', 'STD_5', 'Daily_Return', 'Day_of_Week', 'Month']]
    y = dataFrameInfo[['Close']]
    #Splits stock dataset into 2 parts to train & evaluate model performance
    #Model learns from past data to predict future values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=False)

    #Normalizes features and scales X dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Imports linear regression model & adjusts it to the data being trained
    #Provides predicted stock prices from the test set
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    #Using r2 score to measure predicted accuracy (close to 1 is best)
    r2_test = r2_score(y_test, predictions)
    print("r2 Score: ", r2_test)
    
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    


