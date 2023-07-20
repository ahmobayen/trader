import numpy as np
import pandas as pd
from library import LOOK_BACKS, SCALER


def normalized_split_data(data: pd.DataFrame, features: list[str], is_evalute=False):
    # Choose only Close price of stock
    dataset = data.filter(features).values

    # Scale our data from 0 to 1
    scaled_data = SCALER.transform(dataset) if is_evalute else SCALER.fit_transform(dataset)

    # Train data - 80%, test - 20%
    training_data_len = int(np.ceil(len(dataset) * 0.80))

    # Use our scaled data for training
    x_train, y_train = [], []

    for i in range(LOOK_BACKS, len(scaled_data)):
        x_train.append(scaled_data[i - LOOK_BACKS:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape input data for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if is_evalute:
        print(x_train.shape, y_train.shape)
        return x_train, y_train

    # print(x_train.shape, y_train.shape)
    # return x_train, y_train

    # # Create test dataset
    test_data = scaled_data[training_data_len - LOOK_BACKS:, :]
    x_test = [test_data[i - LOOK_BACKS:i, 0] for i in range(LOOK_BACKS, len(test_data))]
    x_test = np.array(x_test).reshape(-1, LOOK_BACKS, 1)
    y_test = dataset[training_data_len:, :]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test
