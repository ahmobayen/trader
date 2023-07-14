import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library import N_PAST, N_FUTURE, TIME_STEPS


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
    scaler = StandardScaler()

    # Select the relevant columns from the input data
    df_for_training = data[features].astype(float)

    # Perform feature scaling on the training data
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    print(df_for_training_scaled)

    # Split the scaled data into input-output pairs for training
    train_x = []
    train_y = []

    # Creating Time Slots
    for i in range(N_PAST, len(df_for_training_scaled) - N_FUTURE + 1):
        train_x.append(df_for_training_scaled[i - N_PAST:i, 0:df_for_training.shape[1]])
        train_y.append(df_for_training_scaled[i + N_FUTURE - 1:i + N_FUTURE, 0])

    # Convert Time Slots to numpy
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Reshape the training arrays if only one feature is used
    if len(features) == 1:
        train_x = train_x.reshape(-1, 1)
        train_y = train_y.reshape(-1, 1)

    # Print the shape of the training data for debugging purposes
    print(train_x.shape, train_y.shape)

    return train_x, train_y, scaler
