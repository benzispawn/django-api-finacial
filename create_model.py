import requests
import pandas as pd
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Fetch stock data
def fetch_stock_data(symbol, api_key):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }
    response = requests.get('https://www.alphavantage.co/query', params=params)

    if response.status_code == 200:
        data = response.json()
        time_series = data['Time Series (Daily)']

        # Create a DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index', dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns for easier use
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df
    else:
        print("Error fetching data from Alpha Vantage")
        return None


# Preprocess the data
def preprocess_data(df):
    df['close_price'] = df['close']  # Create 'close_price' from 'close'
    df['target'] = df['close_price'].shift(-1)
    df = df.dropna()
    X = df[['close_price']]  # Use 'close_price' as feature
    y = df['target']
    return X, y


# Train the linear regression model
def train_linear_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse}")
    return model


# Save the model
def save_model(model, filename='linear_regression_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


# Main function to run the workflow
def main():
    # Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
    api_key = '43BJ3RR2KY6KTD1Z'
    symbol = 'IBM'

    # Fetch stock data
    df = fetch_stock_data(symbol, api_key)

    if df is not None:
        # Preprocess data
        X, y = preprocess_data(df)

        # Train model
        model = train_linear_regression_model(X, y)

        # Save model
        save_model(model)


if __name__ == '__main__':
    main()
