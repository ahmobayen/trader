from datetime import date

EMA_PERIODS = [8, 20]
TODAY = date.today().strftime('%Y-%m-%d')

N_FUTURE = 10  # Number of days we want to look into the future based on the past days.
N_PAST = 60  # Number of past days we want to use to predict the future.

LOOKBACKS = 15