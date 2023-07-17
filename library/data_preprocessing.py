import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from library import N_PAST, N_FUTURE, LOOKBACKS


def normalized_split_data(data: pd.DataFrame, features: list[str]):
    """
    Splits the data into normalized training samples and corresponding target values.

    Args:
        data (pd.DataFrame): The input data containing features and target variable.
        features (list[str]): List of feature column names.

    Returns:
        tuple: A tuple containing the normalized training samples, target values, and scaler object.

    Raises:
        ValueError: If the length of features is 0 or if the specified feature columns are not found in the data.

    """

    if len(features) == 0:
        raise ValueError("No features provided.")

    # Initialize the StandardScaler object
    scaler = QuantileTransformer(random_state=0)

    # Select the relevant columns from the input data

    # Perform feature scaling on the training data

    # X = data[features].astype(float)
    X = data[features]
    y = data.Target.astype(float)

    if len(features) == 1:
        X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    # Split the scaled data into input-output pairs for training
    train_x = []
    train_y = []

    # Creating Time Slots
    for i in range(N_PAST, len(X_scaled) - N_FUTURE + 1):
        train_x.append(X_scaled[i - N_PAST:i, 0:X_scaled.shape[1]])
        train_y.append(y_scaled[i + N_FUTURE - 1:i + N_FUTURE])

    # Convert Time Slots to numpy
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Reshape the training arrays if only one feature is used
    # if len(features) == 1:
    #     train_x = train_x.reshape(-1, 1)
    #     train_y = train_y.reshape(-1, 1)

    # Print the shape of the training data for debugging purposes
    print(train_x.shape, train_y.shape)

    return train_x, train_y, scaler
