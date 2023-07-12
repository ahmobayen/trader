import pandas as pd
import yfinance as yf
import pandas_ta as ta


from library import EMA_PERIODS, TODAY


def get_data(ticker, start_date='2010-01-01', end_date=TODAY, interval='1d') -> pd.DataFrame:
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
    data = data.reset_index()

    # Calculate Relative Strength Index (RSI)
    data['RSI'] = ta.rsi(data.Close, length=15)

    # Calculate Exponential Moving Averages (EMA)
    for period in EMA_PERIODS:
        data[f'EMA_{period}'] = ta.ema(data.Close, length=period)

    # Calculate price changes
    data['PctChange'] = data.Close.pct_change(periods=-1)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Reset indexing
    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    test_data = get_data(ticker = 'NFLX')
    print(test_data)
