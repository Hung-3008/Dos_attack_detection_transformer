import time
from typing import Dict
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from preprocessor.transformer_preprocessor import prepare_datasets, save_metadata
from .transformer_model import TransformerClassifier


def train_and_evaluate(
    df_train,
    df_test,
    features_to_keep,
    categorical_features,
    device: str = "cpu",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
):
    device = torch.device(device)

    train_ds, test_ds, metadata = prepare_datasets(df_train, df_test, features_to_keep, categorical_features)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_numeric = len(metadata["num_feats"]) if metadata.get("num_feats") is not None else 0
    cat_vocab_sizes = [len(v) for v in metadata["vocabs"].values()]

    model = TransformerClassifier(num_numeric=num_numeric, cat_vocab_sizes=cat_vocab_sizes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop (small, CPU-friendly)
    model.train()
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            num = batch["num"].to(device)
            cat = batch["cat"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(num, cat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

        epoch_loss = epoch_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Evaluate
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            num = batch["num"].to(device)
            cat = batch["cat"].to(device)
            labels = batch["label"].to(device)
            logits = model(num, cat)
            preds = torch.argmax(logits, dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Save model checkpoint and metadata
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(ckpt_dir, f"transformer_{ts}.pth")
    meta_path = os.path.join(ckpt_dir, f"transformer_{ts}_meta.pkl")

    try:
        torch.save(model.state_dict(), model_path)
        # save preprocessing metadata (vocab, scaler)
        save_metadata(metadata, meta_path)
        print(f"Saved transformer checkpoint to {model_path} and metadata to {meta_path}")
    except Exception as e:
        print(f"Warning: failed to save transformer checkpoint: {e}")

    return model, metrics
