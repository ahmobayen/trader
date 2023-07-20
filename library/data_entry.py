import pandas as pd
import yfinance as yf
import pandas_ta as ta


from library import EMA_PERIODS, TODAY


def get_data(ticker, start_date='2010-01-01', end_date=TODAY, interval='1d', is_raw=False) -> pd.DataFrame:
    """
    Retrieves financial data for a specified ticker from Yahoo Finance and performs technical analysis.

    Args:
        ticker (str): Ticker symbol of the financial instrument.
        start_date (str): Start date for data retrieval (default: '2010-01-01').
        end_date (str): End date for data retrieval (default: TODAY).
        interval (str): Interval between data points (default: '1d').

    Returns:
        pd.DataFrame: Dataframe containing the retrieved financial data and calculated technical indicators.

    """

    # Get data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    # Move date from index to column
    # data = data.reset_index()
    data = data.drop(columns=['Adj Close'])

    if is_raw is True:
        data['Shift'] = data.Close.diff(periods=1)
        # data['Target'] = data.Close + data.Shift
        return data

    # Calculate Relative Strength Index (RSI)
    data['RSI'] = ta.rsi(data.Close, length=15)

    # Calculate Exponential Moving Averages (EMA)
    for period in EMA_PERIODS:
        data[f'EMA_{period}'] = ta.ema(data.Close, length=period)

    # Calculate price changes
    data['Shift'] = data.Close.diff(periods=1)

    # Calculate Moving average convergence/divergence (MACD)
    data = pd.concat([data, ta.macd(data.Close)], axis=1)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # data['Target'] = data.Close + data.Shift

    # Reset indexing
    # data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    test_data = get_data(ticker = 'NFLX', start_date='2023-01-02', end_date='2023-02-01',)
    print(test_data)
