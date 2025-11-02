import os
import glob
from datetime import datetime
import html
import sys

# ensure project root is on sys.path so local packages can be imported when running the script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

import gradio as gr

from data_loader.data_loader import load_data
from preprocessor import preprocessor as sk_preproc
from preprocessor.transformer_preprocessor import prepare_datasets, load_metadata
from config import config as project_config

import joblib

try:
    import torch
    from models.transformer_model import TransformerClassifier
except Exception:
    torch = None


CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")


def find_latest_file(pattern):
    files = glob.glob(os.path.join(CHECKPOINT_DIR, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_sklearn_model():
    # prefer RandomForest, then DecisionTree
    joblib_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.joblib"))
    if not joblib_files:
        return None, None
    joblib_files.sort(key=os.path.getmtime, reverse=True)
    model_path = joblib_files[0]
    try:
        model = joblib.load(model_path)
        return model, model_path
    except Exception as e:
        print(f"Failed to load sklearn model {model_path}: {e}")
        return None, None


def load_transformer_model():
    pth = find_latest_file("transformer_*.pth")
    meta = None
    if pth:
        # metadata file uses same timestamp suffix with _meta.pkl
        base = os.path.basename(pth).rsplit(".", 1)[0]
        meta_candidate = os.path.join(CHECKPOINT_DIR, f"{base}_meta.pkl")
        if os.path.exists(meta_candidate):
            meta = meta_candidate
        else:
            # try to find any *_meta.pkl
            meta = find_latest_file("transformer_*_meta.pkl")

    if not pth or not meta:
        return None, None, None

    try:
        metadata = load_metadata(meta)
        num_numeric = len(metadata.get("num_feats", []))
        cat_vocab_sizes = [len(v) for v in metadata.get("vocabs", {}).values()]

        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        model = TransformerClassifier(num_numeric=num_numeric, cat_vocab_sizes=cat_vocab_sizes)
        model.load_state_dict(torch.load(pth, map_location=device))
        model.to(device)
        model.eval()
        return model, metadata, pth
    except Exception as e:
        print(f"Failed to load transformer model: {e}")
        return None, None, None


def run_classification(num_requests: int = 1000, model_choice: str = "auto") -> str:
    # Load data
    df_train, df_test = load_data('data/UNSW_NB15_DoS_train_data.csv', 'data/UNSW_NB15_DoS_test_data.csv')

    n = min(num_requests, len(df_test))
    df_sample = df_test.sample(n=n, random_state=42).reset_index()
    original_indices = df_sample['index'].values

    # Decide model
    chosen = model_choice
    sklearn_model = None
    transformer_model = None
    transformer_meta = None

    if chosen == 'auto' or chosen == 'sklearn':
        sklearn_model, sk_path = load_sklearn_model()
        if sklearn_model is not None:
            chosen = 'sklearn'

    if (chosen == 'auto' or chosen == 'transformer') and sklearn_model is None:
        transformer_model, transformer_meta, tr_path = load_transformer_model()
        if transformer_model is not None:
            chosen = 'transformer'

    if sklearn_model is None and transformer_model is None:
        return "No trained model found in checkpoints/. Train a model first and try again."

    results = []

    if chosen == 'sklearn' and sklearn_model is not None:
        # preprocess full test set to align columns
        X_train, y_train, X_test, y_test = sk_preproc.preprocess_data(
            df_train,
            df_test,
            project_config.FEATURES_TO_KEEP,
            project_config.CATEGORICAL_FEATURES,
        )
        # select sampled rows
        X_sample = X_test.loc[original_indices]
        y_sample = y_test.loc[original_indices]

        y_pred = sklearn_model.predict(X_sample)
        for idx, true, pred in zip(original_indices, y_sample.values, y_pred):
            results.append({'index': int(idx), 'actual': int(true), 'predicted': int(pred)})

    elif chosen == 'transformer' and transformer_model is not None:
        # prepare datasets
        train_ds, test_ds, metadata = prepare_datasets(
            df_train, df_test, project_config.FEATURES_TO_KEEP, project_config.CATEGORICAL_FEATURES
        )
        # pick sample indices
        indices = original_indices
        # create batches
        device = next(transformer_model.parameters()).device
        batch_size = 256
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start+batch_size]
            num_batch = torch.from_numpy(test_ds.num[batch_idx]).to(device)
            cat_batch = torch.from_numpy(test_ds.cat[batch_idx]).to(device)
            with torch.no_grad():
                logits = transformer_model(num_batch, cat_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            trues = test_ds.labels[batch_idx]
            for idx, true, pred in zip(batch_idx, trues, preds):
                results.append({'index': int(idx), 'actual': int(true), 'predicted': int(pred)})

    # Build HTML table with highlights
    df_res = pd.DataFrame(results)
    df_res['actual_name'] = df_res['actual'].map({0: 'Normal', 1: 'DoS'})
    df_res['predicted_name'] = df_res['predicted'].map({0: 'Normal', 1: 'DoS'})
    df_res['correct'] = df_res['actual'] == df_res['predicted']

    # Create HTML with row coloring
    rows = []
    rows.append('<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse; width:100%">')
    rows.append('<thead><tr><th>index</th><th>actual</th><th>predicted</th><th>result</th></tr></thead>')
    rows.append('<tbody>')
    for _, r in df_res.iterrows():
        color = '#d4edda' if r['correct'] else '#f8d7da'
        rows.append(f"<tr style='background:{color}'>")
        rows.append(f"<td>{html.escape(str(r['index']))}</td>")
        rows.append(f"<td>{html.escape(r['actual_name'])}</td>")
        rows.append(f"<td>{html.escape(r['predicted_name'])}</td>")
        rows.append(f"<td>{'OK' if r['correct'] else 'WRONG'}</td>")
        rows.append('</tr>')
    rows.append('</tbody></table>')

    summary = f"<p>Model: {html.escape(chosen)} — Total: {len(df_res)}, Correct: {int(df_res['correct'].sum())}, Incorrect: {int((~df_res['correct']).sum())}</p>"
    return summary + ''.join(rows)


def demo_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# DoS Classification Demo")
        with gr.Row():
            num = gr.Slider(100, 1000, value=1000, step=100, label="Number of requests to classify")
            model_choice = gr.Dropdown(['auto', 'sklearn', 'transformer'], value='auto', label='Model choice')
        run_button = gr.Button('Run Demo')
        output = gr.HTML()

        run_button.click(fn=run_classification, inputs=[num, model_choice], outputs=[output])

    return demo


if __name__ == '__main__':
    app = demo_interface()
    app.launch(server_name='0.0.0.0', server_port=7860, share=False)
