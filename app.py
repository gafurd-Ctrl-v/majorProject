import os
import io
import tempfile
import numpy as np
import torch
import wfdb
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from config import (
    DATA_DIR, CHECKPOINT_PATH, FS, SIGNAL_LEN,
    CLASSES, CLASS_NAMES, CLASS_SUPERCLASS, LEAD_NAMES,
    NUM_CLASSES, BASE_CH, N_HEADS, DROPOUT, TOP_K_LEADS,
)
from dataset          import bandpass_filter, instance_normalize, pad_or_crop
from model            import ECGAttentionNet
from explainability   import GradCAM1D, compute_lead_importance, top_k_leads
from report           import extract_clinical_metrics
from interactive_viz  import render_interactive_ecg
from diagnosis_engine import run_diagnosis_engine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title='ECG Analyzer', layout='wide')


# ── Cached model ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading model…')
def load_model(checkpoint_path: str) -> tuple:
    model = ECGAttentionNet(num_classes=NUM_CLASSES, base_ch=BASE_CH,
                            nhead=N_HEADS, dropout=DROPOUT)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    thresholds = (np.array(ckpt['thresholds'], dtype=np.float32)
                  if 'thresholds' in ckpt
                  else np.full(NUM_CLASSES, 0.5, dtype=np.float32))
    return model.to(DEVICE).eval(), thresholds


@st.cache_resource(show_spinner=False)
def load_grad_cam(_model) -> GradCAM1D:
    return GradCAM1D(_model)


# ── Signal loading ─────────────────────────────────────────────────────────────

def preprocess(signal: np.ndarray) -> np.ndarray:
    signal = pad_or_crop(signal)
    signal = bandpass_filter(signal)
    signal = instance_normalize(signal)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def load_from_ptbxl_path(data_dir: str, record_rel: str) -> np.ndarray:
    full_path = os.path.join(data_dir, record_rel.strip())
    record    = wfdb.rdrecord(full_path)
    return preprocess(record.p_signal.T.astype(np.float32))


def load_from_wfdb_upload(hea_file, dat_file) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        base     = hea_file.name.replace('.hea', '')
        hea_path = os.path.join(tmpdir, hea_file.name)
        dat_path = os.path.join(tmpdir, dat_file.name)
        with open(hea_path, 'wb') as f: f.write(hea_file.read())
        with open(dat_path, 'wb') as f: f.write(dat_file.read())
        record = wfdb.rdrecord(os.path.join(tmpdir, base))
        signal = record.p_signal.T.astype(np.float32)
    return preprocess(signal)


def load_from_csv_upload(csv_file) -> np.ndarray:
    df  = pd.read_csv(csv_file, header=None)
    arr = df.values.astype(np.float32)
    if arr.shape[0] == 12 and arr.shape[1] != 12:
        signal = arr
    elif arr.shape[1] == 12:
        signal = arr.T
    else:
        df2    = pd.read_csv(io.StringIO(csv_file.getvalue().decode()), header=0)
        arr2   = df2.values.astype(np.float32)
        signal = arr2.T if arr2.shape[1] == 12 else arr2
    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got {signal.shape[0]}.")
    return preprocess(signal)


# ── 12-lead matplotlib plot ───────────────────────────────────────────────────

def plot_12_lead(signal:  np.ndarray,
                 heatmap: np.ndarray | None = None,
                 title:   str = '12-Lead ECG') -> Figure:
    fig = Figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=12)
    axes = fig.subplots(6, 2)
    t = np.arange(SIGNAL_LEN) / FS

    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, signal[i], color='#2563eb', linewidth=0.7)
        if heatmap is not None:
            for j in range(0, SIGNAL_LEN - 10, 10):
                alpha = float(heatmap[i][j]) * 0.55
                if alpha > 0.02:
                    ax.axvspan(t[j], t[min(j + 10, SIGNAL_LEN - 1)],
                               alpha=alpha, color='#ef4444', linewidth=0)
        ax.set_title(LEAD_NAMES[i], fontsize=9, loc='left', pad=2)
        ax.set_xlim(0, SIGNAL_LEN / FS)
        ax.set_ylabel('mV', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    return fig


# ── Analysis pipeline ──────────────────────────────────────────────────────────

def run_analysis(model, grad_cam, signal_np: np.ndarray,
                 source_label: str, thresholds: np.ndarray):
    signal_t = torch.FloatTensor(signal_np).unsqueeze(0).to(DEVICE)

    with st.spinner('Running inference…'):
        with torch.no_grad():
            probs = torch.sigmoid(model(signal_t))[0].cpu().numpy()

    predictions = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    top_cls_idx = int(np.argmax(probs))
    top_cls     = CLASSES[top_cls_idx]
    top_conf    = probs[top_cls_idx]
    top_thresh  = float(thresholds[top_cls_idx])

    with st.spinner('Computing Grad-CAM and Attention Rollout…'):
        model.zero_grad()
        heatmap         = grad_cam.generate(signal_t, class_idx=top_cls_idx)
        lead_importance = compute_lead_importance(model, signal_t, top_cls_idx)

    with st.spinner('Extracting clinical intervals…'):
        metrics = extract_clinical_metrics(signal_np)

    sub_diagnoses = run_diagnosis_engine(predictions, lead_importance, metrics)

    # ── Clinical metrics row ──────────────────────────────────────────────────
    is_positive = top_conf >= top_thresh
    st.subheader(
        f'Primary finding: {CLASS_NAMES.get(top_cls, top_cls)} — '
        f'{top_conf * 100:.1f}% confidence'
        + ('' if is_positive else ' (below threshold)')
    )
    st.caption(f'Source: {source_label}  ·  threshold: {top_thresh:.2f}')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Heart Rate',   f"{metrics['hr']} bpm"          if metrics.get('hr')           else '—')
    c2.metric('PR Interval',  f"{metrics['pr_interval']} ms"   if metrics.get('pr_interval')  else '—')
    c3.metric('QRS Duration', f"{metrics['qrs_duration']} ms"  if metrics.get('qrs_duration') else '—')
    c4.metric('QTc',          f"{metrics['qtc']} ms"           if metrics.get('qtc')          else '—')

    st.divider()

    # ── 12-lead ECG plot ──────────────────────────────────────────────────────
    show_cam = st.toggle('Show Grad-CAM overlay on 12-lead plot', value=True)
    st.subheader('12-Lead ECG' + (' + Grad-CAM Overlay' if show_cam else ''))
    fig = plot_12_lead(
        signal_np,
        heatmap if show_cam else None,
        title=f'{CLASS_NAMES.get(top_cls, top_cls)} ({top_conf * 100:.1f}%)  —  {source_label}'
    )
    st.pyplot(fig, use_container_width=True)

    st.divider()

    # ── Interactive ECG visualization ─────────────────────────────────────────
    st.subheader('Interactive ECG Analysis')
    st.caption('Click any lead card to inspect it. Click a territory region to highlight its leads.')

    render_interactive_ecg(
        signal_np       = signal_np,
        heatmap         = heatmap,
        lead_importance = lead_importance,
        predictions     = predictions,
        top_cls         = top_cls,
        sub_diagnoses   = sub_diagnoses[:2],
    )


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    st.title('ECG  Analyzer')
    st.caption('ResNet-1D Style Architecture + Multi-Head | PTB-XL | Grad-CAM')

    with st.sidebar:
        st.header('Configuration')
        checkpoint_path = st.text_input('Model checkpoint', value=CHECKPOINT_PATH)
        data_dir        = st.text_input('PTB-XL root directory', value=DATA_DIR)

        st.divider()
        st.header('Input source')
        input_mode = st.radio(
            'How do you want to load an ECG?',
            options=['PTB-XL record path', 'Upload WFDB files (.hea + .dat)', 'Upload CSV'],
            index=0,
        )

        st.divider()
        st.caption(f'Device: `{DEVICE}`')
        if DEVICE.type == 'cuda':
            free, total = torch.cuda.mem_get_info()
            st.caption(f'VRAM: {free/1e9:.1f} / {total/1e9:.1f} GB free')

    if not os.path.exists(checkpoint_path):
        st.warning(
            f'**No model checkpoint found at `{checkpoint_path}`**\n\n'
            'Train first:\n```\npython train.py\n```'
        )
        st.stop()

    model, thresholds = load_model(checkpoint_path)
    grad_cam          = load_grad_cam(model)

    # ── Mode 1: PTB-XL record path ────────────────────────────────────────────
    if input_mode == 'PTB-XL record path':
        st.subheader('Load from PTB-XL dataset')
        with st.expander('How to find valid record paths'):
            st.code(
                'import os\n'
                f'folder = os.path.join(r"{DATA_DIR}", "records500", "00000")\n'
                'for f in sorted(os.listdir(folder)):\n'
                '    if f.endswith(".hea"):\n'
                '        print("records500/00000/" + f.replace(".hea", ""))',
                language='python'
            )
        record_rel  = st.text_input('Record path (relative, no extension)',
                                    placeholder='records500/00000/00001_hr')
        analyze_btn = st.button('🔍 Analyze ECG', type='primary')
        if analyze_btn:
            if not record_rel.strip():
                st.error('Enter a record path.')
                st.stop()
            with st.spinner('Loading record…'):
                try:
                    signal_np = load_from_ptbxl_path(data_dir, record_rel)
                except FileNotFoundError:
                    st.error(f'Record not found: `{os.path.join(data_dir, record_rel)}.hea`')
                    st.stop()
                except Exception as exc:
                    st.error(f'Load failed: {exc}')
                    st.stop()
            run_analysis(model, grad_cam, signal_np, source_label=record_rel,
                         thresholds=thresholds)

    # ── Mode 2: Upload WFDB ───────────────────────────────────────────────────
    elif input_mode == 'Upload WFDB files (.hea + .dat)':
        st.subheader('Upload a WFDB record')
        st.caption('Both files must share the same base name, e.g. `00001_hr.hea` + `00001_hr.dat`.')
        col1, col2 = st.columns(2)
        with col1:
            hea_file = st.file_uploader('Header file (.hea)', type=['hea'])
        with col2:
            dat_file = st.file_uploader('Signal file (.dat)', type=['dat'])

        if hea_file and dat_file:
            hea_base = hea_file.name.replace('.hea', '')
            dat_base = dat_file.name.replace('.dat', '')
            if hea_base != dat_base:
                st.error(f'Base names must match: `{hea_file.name}` vs `{dat_file.name}`')
                st.stop()
            if st.button('🔍 Analyze ECG', type='primary'):
                with st.spinner('Loading uploaded record…'):
                    try:
                        signal_np = load_from_wfdb_upload(hea_file, dat_file)
                    except Exception as exc:
                        st.error(f'Failed to parse WFDB files: {exc}')
                        st.stop()
                run_analysis(model, grad_cam, signal_np,
                             source_label=f'Uploaded: {hea_file.name}',
                             thresholds=thresholds)
        else:
            st.info('Upload both the `.hea` and `.dat` files to proceed.')

    # ── Mode 3: Upload CSV ────────────────────────────────────────────────────
    elif input_mode == 'Upload CSV':
        st.subheader('Upload a CSV file')
        st.caption('12 leads as columns (rows = samples) or 12 rows (columns = samples). Header row optional.')
        with st.expander('CSV format guide'):
            st.markdown('**Format A — 12 columns (most common):**')
            st.code('I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n0.012,-0.003,...\n...', language='text')
            st.markdown('**Format B — 12 rows:**')
            st.code('0.012,0.015,...   ← Lead I\n-0.003,-0.001,... ← Lead II\n...', language='text')
            st.info('Sampling rate should be 500 Hz (5000 samples = 10s). Other lengths are padded/cropped.')

        csv_file = st.file_uploader('ECG CSV file', type=['csv', 'txt'])
        if csv_file:
            if st.button('🔍 Analyze ECG', type='primary'):
                with st.spinner('Parsing CSV…'):
                    try:
                        signal_np = load_from_csv_upload(csv_file)
                    except Exception as exc:
                        st.error(f'Failed to parse CSV: {exc}')
                        st.stop()
                st.success(
                    f'Loaded: {signal_np.shape[0]} leads × {signal_np.shape[1]} samples '
                    f'({signal_np.shape[1]/FS:.1f}s at {FS}Hz)'
                )
                run_analysis(model, grad_cam, signal_np,
                             source_label=f'Uploaded: {csv_file.name}',
                             thresholds=thresholds)
        else:
            st.info('Upload a CSV file to proceed.')


if __name__ == '__main__':
    main()
