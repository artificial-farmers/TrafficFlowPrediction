"""
Data processing utilities for traffic flow prediction.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_data(train_file: str, test_file: str, lag: int, attr: str = "Lane 1 Flow (Veh/5 Minutes)") -> Tuple:
    """Process data for traffic flow prediction.

    Reshapes and splits train/test data.

    Args:
        train_file: Path to training CSV file
        test_file: Path to test CSV file
        lag: Time lag (number of previous time steps to use as features)
        attr: Column name of the traffic flow attribute to predict

    Returns:
        Tuple containing:
            - X_train: Training features
            - y_train: Training targets
            - X_test: Test features
            - y_test: Test targets
            - scaler: Fitted MinMaxScaler
    """
    # Read and fill missing values
    df_train = pd.read_csv(train_file, encoding="utf-8").fillna(0)
    df_test = pd.read_csv(test_file, encoding="utf-8").fillna(0)

    # Scale data to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[attr].values.reshape(-1, 1))

    # Transform training and test data
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Create sequences with specified lag
    train_sequences, test_sequences = [], []

    for i in range(lag, len(flow_train)):
        train_sequences.append(flow_train[i - lag: i + 1])

    for i in range(lag, len(flow_test)):
        test_sequences.append(flow_test[i - lag: i + 1])

    # Convert to numpy arrays
    train_sequences = np.array(train_sequences)
    test_sequences = np.array(test_sequences)

    # Shuffle training data
    np.random.shuffle(train_sequences)

    # Split into features (X) and targets (y)
    X_train = train_sequences[:, :-1]
    y_train = train_sequences[:, -1]
    X_test = test_sequences[:, :-1]
    y_test = test_sequences[:, -1]

    return X_train, y_train, X_test, y_test, scaler
