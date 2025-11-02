import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, num_array: np.ndarray, cat_array: np.ndarray, labels: np.ndarray):
        self.num = num_array.astype(np.float32)
        self.cat = cat_array.astype(np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "num": torch.from_numpy(self.num[idx]),
            "cat": torch.from_numpy(self.cat[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_categorical_vocabs(df: pd.DataFrame, categorical_features: List[str]) -> Dict[str, Dict[str, int]]:
    """Build vocab maps for categorical features. Reserve 0 for UNK."""
    vocabs = {}
    for feat in categorical_features:
        vals = pd.Series(df[feat].fillna("__NA__")).astype(str).unique().tolist()
        # start indexing at 1 so 0 can be UNK
        vocab = {v: i + 1 for i, v in enumerate(vals)}
        vocabs[feat] = vocab
    return vocabs


def transform_categorical(df: pd.DataFrame, categorical_features: List[str], vocabs: Dict[str, Dict[str, int]]) -> np.ndarray:
    arr = np.zeros((len(df), len(categorical_features)), dtype=np.int64)
    for j, feat in enumerate(categorical_features):
        col = df[feat].fillna("__NA__").astype(str).tolist()
        vocab = vocabs[feat]
        # map to id, unknowns -> 0
        arr[:, j] = [vocab.get(v, 0) for v in col]
    return arr


def fit_numeric_scaler(df: pd.DataFrame, numeric_features: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[numeric_features].fillna(0.0).values)
    return scaler


def transform_numeric(df: pd.DataFrame, numeric_features: List[str], scaler: StandardScaler) -> np.ndarray:
    arr = scaler.transform(df[numeric_features].fillna(0.0).values)
    return arr.astype(np.float32)


def prepare_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features_to_keep: List[str],
    categorical_features: List[str],
    label_col: str = "label",
):
    """
    Fit vocabs and scalers on training data then transform both train/test into PyTorch datasets.

    Returns: (train_dataset, test_dataset, metadata) where metadata contains vocabs and scaler
    """
    # split numeric and categorical features according to categorical_features
    cat_feats = [f for f in features_to_keep if f in categorical_features]
    num_feats = [f for f in features_to_keep if f not in categorical_features]

    # Build vocabs and scalers on training set
    vocabs = build_categorical_vocabs(df_train, cat_feats)
    scaler = fit_numeric_scaler(df_train, num_feats) if num_feats else None

    # Transform
    X_train_cat = transform_categorical(df_train, cat_feats, vocabs) if cat_feats else np.zeros((len(df_train), 0), dtype=np.int64)
    X_test_cat = transform_categorical(df_test, cat_feats, vocabs) if cat_feats else np.zeros((len(df_test), 0), dtype=np.int64)

    X_train_num = transform_numeric(df_train, num_feats, scaler) if num_feats else np.zeros((len(df_train), 0), dtype=np.float32)
    X_test_num = transform_numeric(df_test, num_feats, scaler) if num_feats else np.zeros((len(df_test), 0), dtype=np.float32)

    y_train = df_train[label_col].values.astype(np.int64)
    y_test = df_test[label_col].values.astype(np.int64)

    train_ds = TabularDataset(X_train_num, X_train_cat, y_train)
    test_ds = TabularDataset(X_test_num, X_test_cat, y_test)

    metadata = {
        "cat_feats": cat_feats,
        "num_feats": num_feats,
        "vocabs": vocabs,
        "scaler": scaler,
    }

    return train_ds, test_ds, metadata


def save_metadata(metadata: dict, path: str):
    # Note: scaler is picklable; vocabs are dicts
    with open(path, "wb") as f:
        pickle.dump(metadata, f)


def load_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
