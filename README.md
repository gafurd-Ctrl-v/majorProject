# ECG Attention Analyzer

ResNet-1D + Multi-Head Attention model for 12-lead ECG classification on PTB-XL,
with Grad-CAM temporal heatmaps, attention-rollout lead importance, and
an AI-generated clinical summary via the Anthropic API.

---

## Configuration — start here

**All settings live in `config.py`.** You should never need to edit any other file
just to change a path or hyperparameter.

The most common things you'll change:

```python
# config.py

DATA_DIR        = '/data/ptbxl'         # ← point this at your PTB-XL folder
CHECKPOINT_DIR  = 'checkpoints'         # ← where best_model.pt is saved
BATCH_SIZE      = 16                    # ← lower to 8 if CUDA out-of-memory
EPOCHS          = 30
LEARNING_RATE   = 3e-4
NUM_WORKERS     = 4                     # ← set to 0 on Windows
ANTHROPIC_API_KEY = None                # ← or paste your key here
```

---

## Project structure

```
ecg_model/
├── config.py           ← SINGLE SOURCE OF TRUTH — edit this file only
├── dataset.py          PTB-XL loading, filtering, normalisation
├── model.py            ECGAttentionNet architecture
├── explainability.py   Grad-CAM + Attention Rollout
├── report.py           NeuroKit2 intervals + Anthropic API report
├── train.py            Training loop (AMP, OneCycleLR)
├── app.py              Streamlit dashboard
└── requirements.txt
```

---

## Setup

```bash
# 1. Activate your virtual environment
source ecg_env/bin/activate      # Linux/Mac
ecg_env\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.py — set DATA_DIR to your PTB-XL path
```

---

## Training

```bash
# Uses all values from config.py
python train.py

# Override any value on the command line (takes precedence over config.py)
python train.py --data_dir /other/path --batch_size 8 --epochs 50
```

---

## Streamlit dashboard

```bash
streamlit run app.py
```

The sidebar pre-fills with `DATA_DIR` and `CHECKPOINT_PATH` from `config.py`.

---

## Architecture

================================================================================
                          DIMENSION FLOW THROUGH MODEL
================================================================================

Step                           Operation                           Shape
-------------------------------------------------------------------------------------
1                              Input                               (1, 12, 5000)
2                              Reshape (combine batch & leads)     (12, 1, 5000)
3                              Backbone CNN                        (12, 512, 79)       
4                              Global avg pooling (over time)      (12, 512)
5                              Reshape back (separate leads)       (1, 12, 512)
6                              Add positional encoding             (1, 12, 512)
7.1                            Attention layer 1                   (1, 12, 512)        
7.2                            Attention layer 2                   (1, 12, 512)        
8                              Mean over leads dimension           (1, 512)
9                              Classification head                 (1, 27)

✅ Final output shape: (1, 27)
   (Batch size, 27 classes)
================================================================================

~2.1 M parameters total.

---

## What each config value controls

| Variable | Used in | Effect |
|---|---|---|
| `DATA_DIR` | dataset.py, train.py, app.py | Where PTB-XL lives |
| `CHECKPOINT_DIR` / `CHECKPOINT_PATH` | train.py, app.py | Where model is saved/loaded |
| `FS` | dataset.py, report.py, app.py | Sampling rate (Hz) |
| `SIGNAL_LEN` | dataset.py, explainability.py, app.py | Samples per recording |
| `CLASSES` | dataset.py, train.py, report.py, app.py | Label names |
| `LEAD_NAMES` | explainability.py, report.py, app.py | Lead labels |
| `LEAD_II_INDEX` | report.py | Which lead NeuroKit2 analyses |
| `FILTER_LOWCUT/HIGHCUT/ORDER` | dataset.py | Bandpass filter settings |
| `TEST_FOLD` / `VAL_FOLD` | dataset.py | Train/val/test split |
| `BASE_CH` | model.py, train.py, app.py | Channel width (scales VRAM) |
| `N_HEADS` | model.py, train.py, app.py | Attention heads |
| `DROPOUT` | model.py, train.py, app.py | Regularisation |
| `BATCH_SIZE` | dataset.py, train.py | Samples per step |
| `EPOCHS` | train.py | Training length |
| `LEARNING_RATE` | train.py | Peak LR for OneCycleLR |
| `GRAD_CLIP_NORM` | train.py | Max gradient norm |
| `NUM_WORKERS` | dataset.py, train.py | DataLoader workers |
| `TOP_K_LEADS` | explainability.py, app.py | Leads highlighted in chart |
| `ANTHROPIC_MODEL` | report.py | Claude model for reports |
| `ANTHROPIC_MAX_TOKENS` | report.py | Max tokens per report |
| `ANTHROPIC_API_KEY` | report.py | API key (or use env var) |
