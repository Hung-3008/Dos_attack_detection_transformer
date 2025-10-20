import pandas as pd
import numpy as np

def preprocess_data(df_train, df_test, features_to_keep, categorical_features):
    """
    Preprocesses the training and testing data.

    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_test (pd.DataFrame): The testing DataFrame.
        features_to_keep (list): A list of feature names to keep.
        categorical_features (list): A list of categorical feature names.

    Returns:
        tuple: A tuple containing the preprocessed X_train, y_train, X_test, y_test.
    """
    # Select the features and target
    X_train = df_train[features_to_keep].copy()
    y_train = df_train['label']
    X_test = df_test[features_to_keep].copy()
    y_test = df_test['label']

    # Apply one-hot encoding to categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_features, dummy_na=True)
    X_test = pd.get_dummies(X_test, columns=categorical_features, dummy_na=True)

    # Align columns between training and testing sets
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Handle potential infinite values if any
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    return X_train, y_train, X_test, y_test
