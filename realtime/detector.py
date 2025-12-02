import torch
import pickle
import numpy as np
import os
import sys

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import TransformerClassifier
from preprocessor.transformer_preprocessor import transform_numeric, transform_categorical

class Detector:
    def __init__(self, model_path, meta_path, device="cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        print(f"[Detector] Loading model from {self.model_path}...")
        
        # Load metadata
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        num_numeric = len(self.metadata["num_feats"]) if self.metadata.get("num_feats") else 0
        cat_vocab_sizes = [len(v) for v in self.metadata["vocabs"].values()]
        
        # Initialize model
        self.model = TransformerClassifier(num_numeric=num_numeric, cat_vocab_sizes=cat_vocab_sizes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("[Detector] Model loaded successfully.")

    def predict(self, feature_df):
        """
        Predict DoS for a DataFrame of features.
        Returns a list of (prediction, confidence) tuples.
        """
        if feature_df.empty:
            return []

        cat_feats = self.metadata["cat_feats"]
        num_feats = self.metadata["num_feats"]
        vocabs = self.metadata["vocabs"]
        scaler = self.metadata["scaler"]

        # Transform features
        X_cat = transform_categorical(feature_df, cat_feats, vocabs) if cat_feats else np.zeros((len(feature_df), 0), dtype=np.int64)
        X_num = transform_numeric(feature_df, num_feats, scaler) if num_feats else np.zeros((len(feature_df), 0), dtype=np.float32)

        # To Tensor
        num_tensor = torch.from_numpy(X_num).to(self.device)
        cat_tensor = torch.from_numpy(X_cat).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(num_tensor, cat_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
        results = []
        for i in range(len(preds)):
            label = preds[i].item() # 0 or 1
            conf = probs[i][label].item()
            results.append((label, conf))
            
        return results
