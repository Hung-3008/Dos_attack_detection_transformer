"""Gradio demo 2

Features:
- Limits total submissions to 1000
- For each submission: shows request detail immediately, runs model (shows loading), then displays result
- Tracks stats and when 1000 submissions reached shows a summary table and disables further submissions
"""
import os
import sys
import time
import random
import html
import glob
from datetime import datetime

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
REQUEST_LIMIT = 1000


def find_latest_file(pattern):
    files = glob.glob(os.path.join(CHECKPOINT_DIR, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_sklearn_model():
    joblib_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.joblib"))
    if not joblib_files:
        return None, None
    joblib_files.sort(key=os.path.getmtime, reverse=True)
    path = joblib_files[0]
    try:
        return joblib.load(path), path
    except Exception:
        return None, None


def load_transformer_model():
    pth = find_latest_file("transformer_*.pth")
    if not pth:
        return None, None, None
    base = os.path.basename(pth).rsplit('.', 1)[0]
    meta = os.path.join(CHECKPOINT_DIR, f"{base}_meta.pkl")
    if not os.path.exists(meta):
        meta = find_latest_file("transformer_*_meta.pkl")
    if not meta:
        return None, None, None

    try:
        metadata = load_metadata(meta)
        num_numeric = len(metadata.get('num_feats', []))
        cat_vocab_sizes = [len(v) for v in metadata.get('vocabs', {}).values()]
        device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        model = TransformerClassifier(num_numeric=num_numeric, cat_vocab_sizes=cat_vocab_sizes)
        model.load_state_dict(torch.load(pth, map_location=device))
        model.to(device)
        model.eval()
        return model, metadata, pth
    except Exception:
        return None, None, None


# Load data and preprocessing once at startup
df_train, df_test = load_data('data/UNSW_NB15_DoS_train_data.csv', 'data/UNSW_NB15_DoS_test_data.csv')

# sklearn preprocessing (for sklearn models)
try:
    X_train_skl, y_train_skl, X_test_skl, y_test_skl = sk_preproc.preprocess_data(
        df_train, df_test, project_config.FEATURES_TO_KEEP, project_config.CATEGORICAL_FEATURES
    )
except Exception:
    # fallback: naive selection
    X_train_skl = X_test_skl = pd.DataFrame()
    y_train_skl = y_test_skl = pd.Series([])

# transformer preprocessing
try:
    train_ds, test_ds, transformer_meta = prepare_datasets(
        df_train, df_test, project_config.FEATURES_TO_KEEP, project_config.CATEGORICAL_FEATURES
    )
except Exception:
    train_ds = test_ds = transformer_meta = None

# Load models (if available)
sklearn_model, sk_path = load_sklearn_model()
transformer_model, transformer_meta_loaded, tr_path = load_transformer_model()


def make_request_html(row: pd.Series) -> str:
    # compact single-line inline display: key=value, ... (truncated)
    pairs = [f"{html.escape(str(k))}={html.escape(str(v))}" for k, v in row.items()]
    line = ", ".join(pairs)
    if len(line) > 800:
        display_line = line[:800] + "..."
    else:
        display_line = line
    # show as single-line with tooltip containing full text
    return f"<div style='background:#111;color:#fff;padding:6px;border-radius:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' title='{html.escape(line)}'>{display_line}</div>"


def make_result_html(pred, true=None, prob=None) -> str:
    pred_name = 'DoS' if int(pred) == 1 else 'Normal'
    true_part = f"<p>True: {'DoS' if int(true)==1 else 'Normal'}</p>" if true is not None else ''
    prob_part = f"<p>Confidence: {prob:.3f}</p>" if prob is not None else ''
    color = '#28a745' if true is None or int(pred) == int(true) else '#dc3545'
    return f"<div style='background:#000;color:#fff;padding:8px;border-left:6px solid {color};border-radius:6px'><p><strong>Predicted:</strong> {pred_name}</p>{true_part}{prob_part}</div>"


def make_summary_html(state: dict) -> str:
    total = state['count']
    total_time = state['total_time']
    avg = total_time / total if total > 0 else 0.0
    success = state['correct']
    rate = success / total if total > 0 else 0.0
    rows = [f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in [
        ('Total requests', total),
        ('Total time (s)', f"{total_time:.3f}"),
        ('Avg time/request (s)', f"{avg:.4f}"),
        ('Success count', success),
        ('Success rate', f"{rate:.3%}"),
    ]]
    return f"<table border='1' style='border-collapse:collapse;color:#fff;background:#111;padding:8px'><tbody>{''.join(rows)}</tbody></table>"


def make_current_row_html(idx: int, row: pd.Series) -> str:
    """Render a single-line table row showing the request properties compactly."""
    # compact description (single cell) with tooltip
    pairs = [f"{html.escape(str(k))}={html.escape(str(v))}" for k, v in row.items()]
    desc = ", ".join(pairs)
    if len(desc) > 600:
        short = desc[:600] + "..."
    else:
        short = desc

    return (
        "<table style='width:100%;border-collapse:collapse;color:#fff;background:#111'>"
        "<tr>"
        f"<td style='padding:6px;border:1px solid #222;width:80px'><strong>#{idx}</strong></td>"
        f"<td style='padding:6px;border:1px solid #222;white-space:nowrap;overflow:hidden;text-overflow:ellipsis' title='{html.escape(desc)}'>{short}</td>"
        "</tr>"
        "</table>"
    )


def make_history_table_html(history: list) -> str:
    """Render a compact history table from a list of dict entries."""
    if not history:
        return "<div style='color:#fff'>No processed requests yet</div>"
    header = (
        "<tr style='background:#222;color:#fff'>"
        "<th style='padding:6px;border:1px solid #333'>#</th>"
        "<th style='padding:6px;border:1px solid #333'>idx</th>"
        "<th style='padding:6px;border:1px solid #333'>actual</th>"
        "<th style='padding:6px;border:1px solid #333'>pred</th>"
        "<th style='padding:6px;border:1px solid #333'>time(s)</th>"
        "<th style='padding:6px;border:1px solid #333'>conf</th>"
        "<th style='padding:6px;border:1px solid #333'>result</th>"
        "</tr>"
    )
    rows = []
    for i, e in enumerate(history, start=1):
        rows.append(
            "<tr>"
            f"<td style='padding:6px;border:1px solid #333'>{i}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(str(e.get('idx','')))}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(str(e.get('actual','')))}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(str(e.get('pred','')))}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(format(e.get('time', 0.0), '.3f'))}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(str(round(e.get('prob',0.0),3)) if e.get('prob') is not None else '')}</td>"
            f"<td style='padding:6px;border:1px solid #333'>{html.escape(str(e.get('result','')))}</td>"
            "</tr>"
        )
    return f"<table style='width:100%;border-collapse:collapse;color:#fff'>{header}{''.join(rows)}</table>"


def classifier_generator(state: dict, max_requests: int):
    # sequentially process requests up to max_requests with 1s delay between
    # normalize state dict (gr.State may provide a dict missing some keys)
    if state is None:
        state = {}
    # if state is a gr.State object wrapper, try to extract value
    if not isinstance(state, dict) and hasattr(state, 'value'):
        state = state.value or {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault('count', 0)
    state.setdefault('start_time', time.time())
    state.setdefault('total_time', 0.0)
    state.setdefault('correct', 0)
    state.setdefault('log_html', '')
    # accumulated current-request log (will be appended to each iteration)
    state.setdefault('current_request_log_html', '')
    state.setdefault('history', [])

    # disable the send button while running
    history_html = make_history_table_html(state.get('history', []))
    # outputs mapping: detail_out (history_html), current_request_display (accumulated current log),
    # result_out (starting / final summary), send_btn update, state
    yield (history_html, state['current_request_log_html'], '<div style="color:#fff">Starting...</div>', gr.update(interactive=False), state)

    while state['count'] < max_requests:
        idx = random.randrange(len(df_test))
        row = df_test.iloc[idx]
        # current request displayed as a single-line table row
        current_html = make_current_row_html(idx, row)

        # show current request immediately (history stays same)
        history_html = make_history_table_html(state.get('history', []))
        # append current_html to the accumulated log and send it to the current_request_display;
        # keep result_out empty during processing
        state['current_request_log_html'] += current_html
        yield (history_html, state['current_request_log_html'], '', gr.update(interactive=False), state)
        # ensure the current request detail stays visible for a short time before running the model
        time.sleep(0.5)

        # run model
        t0 = time.time()
        pred = None
        prob = None
        true_label = int(row['label']) if 'label' in row.index else None
        try:
            if transformer_model is not None:
                device = next(transformer_model.parameters()).device
                num_tensor = torch.from_numpy(test_ds.num[[idx]]).to(device)
                cat_tensor = torch.from_numpy(test_ds.cat[[idx]]).to(device)
                with torch.no_grad():
                    logits = transformer_model(num_tensor, cat_tensor)
                    pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                    try:
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        prob = float(probs.max())
                    except Exception:
                        prob = None
            elif sklearn_model is not None:
                Xs = X_test_skl.iloc[[idx]]
                pred = int(sklearn_model.predict(Xs)[0])
                if hasattr(sklearn_model, 'predict_proba'):
                    prob = float(sklearn_model.predict_proba(Xs)[0].max())
            else:
                pred = None
        except Exception:
            pred = None

        elapsed = time.time() - t0
        state['count'] += 1
        state['total_time'] += elapsed
        if pred is not None and true_label is not None and int(pred) == int(true_label):
            state['correct'] += 1

        pred_name = 'DoS' if pred == 1 else ('Normal' if pred == 0 else 'Unknown')
        result_flag = 'OK' if (pred is not None and true_label is not None and int(pred) == int(true_label)) else 'WRONG'

        # append to history and render updated history table
        state['history'].append({'idx': idx, 'actual': ('DoS' if true_label==1 else 'Normal' if true_label==0 else 'NA'), 'pred': pred_name, 'time': elapsed, 'prob': prob or 0.0, 'result': result_flag})
        history_html = make_history_table_html(state.get('history', []))

        # show updated history; keep accumulated current-request log visible (do not clear)
        # keep result_out empty until the final summary
        yield (history_html, state['current_request_log_html'], '', gr.update(interactive=False), state)

        # 1 second delay between requests
        time.sleep(1)

    # done, show summary and keep button disabled
    summary = make_summary_html(state)
    history_html = make_history_table_html(state.get('history', []))
    # final mapping: history -> detail_out, current (accumulated) -> current_request_display, summary -> result_out
    yield (history_html, state['current_request_log_html'], summary, gr.update(interactive=False), state)


def build_app():
    with gr.Blocks(theme=gr.themes.Default()) as demo:
        gr.Markdown("# DoS Demo (limited to 1000 requests)")
        with gr.Row():
            with gr.Column():
                send_btn = gr.Button("Send Request", elem_id='send_req_btn')
                max_requests = gr.Number(value=1000, label="Max requests (global limit)", precision=0)
                # new HTML component (displayed in the first column, directly below max_requests)
                current_request_display = gr.HTML('<div style="color:#fff">Current Request: None</div>')
            detail_out = gr.HTML('<div style="color:#fff">Request detail will appear here</div>')
            result_out = gr.HTML('<div style="color:#fff">Result will appear here</div>')


        # state to keep counts and stats
        state = gr.State({'count': 0, 'start_time': time.time(), 'total_time': 0.0, 'correct': 0})

        # wire button: outputs are detail_out (history), current_request_display (current request),
        # result_out (starting message / final summary), send_btn (to allow disabling), state
        send_btn.click(
            classifier_generator,
            inputs=[state, max_requests],
            outputs=[detail_out, current_request_display, result_out, send_btn, state],
        )

    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch(server_name='0.0.0.0', server_port=7861, share=False)
