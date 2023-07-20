import numpy as np
import pandas as pd

from library import LOOK_BACKS, SCALER

def make_prediction(data, model):
    def predict(num_prediction, model):
        prediction_list = close_data[-LOOK_BACKS:]

        for _ in range(num_prediction):
            x = prediction_list[-LOOK_BACKS:]
            x_scaled = SCALER.transform(x.reshape(-1, 1))
            x_scaled = x_scaled.reshape((1, LOOK_BACKS, 1))
            out = model.predict(x_scaled)[0][0]
            prediction_list = np.append(prediction_list, SCALER.inverse_transform(out.reshape(-1, 1)))
        prediction_list = prediction_list[LOOK_BACKS - 1:]

        return prediction_list

    def predict_dates(num_prediction):
        last_date = data.index.values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        return prediction_dates

    close_data = data.Close.values.reshape((-1))

    num_prediction = 15
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    result = pd.concat([pd.DataFrame(forecast_dates), pd.DataFrame(forecast)],ignore_index=True, axis=1)
    return result.rename({0: 'Date', 1: 'Close'}, axis=1)
