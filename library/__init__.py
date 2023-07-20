from datetime import date
from sklearn.preprocessing import MinMaxScaler

EMA_PERIODS = [8, 20]
TODAY = date.today().strftime('%Y-%m-%d')

SCALER = MinMaxScaler(feature_range=(0, 1))
LOOK_BACKS = 60