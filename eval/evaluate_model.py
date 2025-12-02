import sys
import os
import glob
import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer_model import TransformerClassifier
from models import train_transformer
from preprocessor.transformer_preprocessor import load_metadata, transform_categorical, transform_numeric, TabularDataset
from config import config

def load_latest_checkpoint(checkpoint_dir):
    # Prioritize the original trained model
    original_ckpt = os.path.join(checkpoint_dir, "transformer_20251102_210010.pth")
    if os.path.exists(original_ckpt):
        meta_path = original_ckpt.replace(".pth", "_meta.pkl")
        if os.path.exists(meta_path):
            return original_ckpt, meta_path

    # Fallback to latest checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "transformer_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    
    # Find corresponding metadata
    meta_path = latest_ckpt.replace(".pth", "_meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found for {latest_ckpt}")
        
    return latest_ckpt, meta_path

def map_columns(df, dataset_name):
    """
    Map columns from different datasets to the expected features.
    """
    mapped_df = pd.DataFrame()
    
    # Expected features
    features = config.FEATURES_TO_KEEP
    
    if dataset_name == "UNSW_NB15":
        return df
        
    elif dataset_name == "Botnet":
        # Direct mappings
        if 'proto' in df.columns: mapped_df['proto'] = df['proto']
        if 'state_number' in df.columns: mapped_df['state'] = df['state_number']
        if 'srate' in df.columns: mapped_df['sload'] = df['srate']
        if 'drate' in df.columns: mapped_df['dload'] = df['drate']
        if 'mean' in df.columns: mapped_df['smean'] = df['mean']
        
        # Fill missing features with 0
        for feat in features:
            if feat not in mapped_df.columns:
                mapped_df[feat] = 0
                
        # Label
        if 'attack' in df.columns:
            mapped_df['label'] = df['attack']
        
    elif dataset_name == "TonIOT":
        mapping = {
            'duration': 'dur',
            'src_pkts': 'spkts',
            'dst_pkts': 'dpkts',
            'src_bytes': 'sbytes',
            'dst_bytes': 'dbytes',
            'proto': 'proto',
            'service': 'service',
            'conn_state': 'state',
            'label': 'label'
        }
        
        for original, target in mapping.items():
            if original in df.columns:
                mapped_df[target] = df[original]
                
        # Fill missing features with 0
        for feat in features:
            if feat not in mapped_df.columns:
                mapped_df[feat] = 0
                
        # Ensure label is present
        if 'label' not in mapped_df.columns and 'label' in df.columns:
             mapped_df['label'] = df['label']

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    # Ensure all expected features are present
    for feat in features:
        if feat not in mapped_df.columns:
            mapped_df[feat] = 0
            
    # Ensure label is present
    if 'label' not in mapped_df.columns:
        if 'label' in df.columns:
            mapped_df['label'] = df['label']
        elif 'attack' in df.columns:
             mapped_df['label'] = df['attack']
        else:
             raise ValueError("Label column not found")
             
    return mapped_df

def evaluate_existing_model(model, metadata, df, dataset_name, device):
    print(f"\nEvaluating pre-trained model on {dataset_name}...")
    
    # Map columns
    df_mapped = map_columns(df, dataset_name)
    
    # Preprocess
    cat_feats = metadata["cat_feats"]
    num_feats = metadata["num_feats"]
    vocabs = metadata["vocabs"]
    scaler = metadata["scaler"]
    
    # Transform
    X_cat = transform_categorical(df_mapped, cat_feats, vocabs) if cat_feats else np.zeros((len(df_mapped), 0), dtype=np.int64)
    X_num = transform_numeric(df_mapped, num_feats, scaler) if num_feats else np.zeros((len(df_mapped), 0), dtype=np.float32)
    y = df_mapped['label'].values.astype(np.int64)
    
    dataset = TabularDataset(X_num, X_cat, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Inference
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            num = batch["num"].to(device)
            cat = batch["cat"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(num, cat)
            preds = torch.argmax(logits, dim=1)
            
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print(f"F1 Score: {f1:.4f}")
    
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist()
    }

def perform_cross_validation(df, dataset_name, device, n_splits=5):
    print(f"\nPerforming {n_splits}-fold Cross-Validation on {dataset_name}...")
    
    # Map columns
    df_mapped = map_columns(df, dataset_name)
    
    # Check class distribution
    label_counts = df_mapped['label'].value_counts()
    print(f"Class distribution:\n{label_counts}")
    
    if len(label_counts) < 2:
        print(f"Skipping CV for {dataset_name}: Only one class present.")
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    total_cm = None
    f1_scores = []
    precisions = []
    recalls = []
    
    fold = 1
    for train_index, test_index in skf.split(df_mapped, df_mapped['label']):
        print(f"Fold {fold}/{n_splits}")
        
        df_train = df_mapped.iloc[train_index]
        df_test = df_mapped.iloc[test_index]
        
        # Train and Evaluate
        model, metrics = train_transformer.train_and_evaluate(
            df_train,
            df_test,
            config.FEATURES_TO_KEEP,
            config.CATEGORICAL_FEATURES,
            device=device,
            epochs=5, 
            batch_size=64,
            lr=3e-4
        )
        
        cm = metrics['confusion_matrix']
        
        if total_cm is None:
            total_cm = cm
        else:
            # Ensure shapes match
            if total_cm.shape == cm.shape:
                total_cm += cm
            else:
                # Pad if necessary (e.g. if one fold misses a class)
                pass
        
        report = metrics['classification_report']
        f1_scores.append(report['weighted avg']['f1-score'])
        precisions.append(report['weighted avg']['precision'])
        recalls.append(report['weighted avg']['recall'])
        
        fold += 1
        
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    return {
        "f1": float(avg_f1),
        "precision": float(avg_precision),
        "recall": float(avg_recall),
        "confusion_matrix": total_cm.tolist() if total_cm is not None else []
    }

def get_balanced_sample(df, n=1000, label_col='label'):
    """
    Sample n items, attempting to balance classes (50/50).
    """
    try:
        # If n is larger than dataset, return all
        if len(df) < n:
            return df
            
        # Check if we have multiple classes
        if df[label_col].nunique() > 1:
            # Target samples per class
            n_per_class = n // 2
            
            # Sample from each class
            sampled_dfs = []
            for label, group in df.groupby(label_col):
                # If group is smaller than target, take all
                if len(group) < n_per_class:
                    sampled_dfs.append(group)
                else:
                    sampled_dfs.append(group.sample(n=n_per_class, random_state=42))
            
            return pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            return df.sample(n=n, random_state=42)
    except Exception as e:
        print(f"Sampling failed: {e}. Returning random sample.")
        return df.sample(n=n, random_state=42)

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    output_dir = os.path.join(base_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}
    
    # 1. Evaluate on UNSW_NB15_testing-set.csv
    try:
        ckpt_path, meta_path = load_latest_checkpoint(checkpoint_dir)
        metadata = load_metadata(meta_path)
        
        num_numeric = len(metadata["num_feats"]) if metadata.get("num_feats") else 0
        cat_vocab_sizes = [len(v) for v in metadata["vocabs"].values()]
        
        model = TransformerClassifier(num_numeric=num_numeric, cat_vocab_sizes=cat_vocab_sizes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        
        print("Loading UNSW_NB15_testing-set.csv...")
        df_unsw = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing-set.csv'))
        res = evaluate_existing_model(model, metadata, df_unsw, "UNSW_NB15", device)
        results["UNSW_NB15"] = res
    except Exception as e:
        print(f"Error evaluating UNSW_NB15: {e}")
    
    # 2. Cross-Validation on Botnet
    try:
        print("Loading Botnet_Final_10_best_Training.csv...")
        df_botnet = pd.read_csv(os.path.join(data_dir, 'Botnet_Final_10_best_Training.csv'))
        # Ensure we have 'attack' column for stratification
        if 'attack' in df_botnet.columns:
             df_botnet_sample = get_balanced_sample(df_botnet, n=1000, label_col='attack')
        else:
             df_botnet_sample = df_botnet.sample(n=1000, random_state=42)
             
        res = perform_cross_validation(df_botnet_sample, "Botnet", device)
        if res:
            results["Botnet"] = res
    except Exception as e:
        print(f"Error evaluating Botnet dataset: {e}")

    # 3. Cross-Validation on TonIOT
    try:
        print("Loading TonIOT_train_test_network.csv...")
        df_toniot = pd.read_csv(os.path.join(data_dir, 'TonIOT_train_test_network.csv'))
        if 'label' in df_toniot.columns:
            df_toniot_sample = get_balanced_sample(df_toniot, n=1000, label_col='label')
        else:
            df_toniot_sample = df_toniot.sample(n=1000, random_state=42)
            
        res = perform_cross_validation(df_toniot_sample, "TonIOT", device)
        if res:
            results["TonIOT"] = res
    except Exception as e:
        print(f"Error evaluating TonIOT dataset: {e}")
        
    # Save results
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {json_path}")

if __name__ == "__main__":
    main()
