import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_DIR, FS, SIGNAL_LEN, CLASSES,
    FILTER_LOWCUT, FILTER_HIGHCUT, FILTER_ORDER,
    TEST_FOLD, VAL_FOLD, BATCH_SIZE, NUM_WORKERS,
)

# Augmentation config (safe-import so an older config.py still works)
def _cfg(name, default):
    try:
        import config as _c
        return getattr(_c, name)
    except AttributeError:
        return default

AUG_TIME_SHIFT        = _cfg('AUG_TIME_SHIFT',        200)
AUG_AMPLITUDE_SCALE   = _cfg('AUG_AMPLITUDE_SCALE',   0.25)
AUG_NOISE_MIN         = _cfg('AUG_NOISE_MIN',         0.005)
AUG_NOISE_MAX         = _cfg('AUG_NOISE_MAX',         0.02)
AUG_LEAD_DROPOUT_PROB = _cfg('AUG_LEAD_DROPOUT_PROB', 0.3)


# ── Signal preprocessing ──────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    nyq  = FS / 2.0
    b, a = butter(FILTER_ORDER, [FILTER_LOWCUT / nyq, FILTER_HIGHCUT / nyq], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def instance_normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1,  keepdims=True) + 1e-8
    return (signal - mean) / std


def pad_or_crop(signal: np.ndarray) -> np.ndarray:
    T = signal.shape[1]
    if T > SIGNAL_LEN:
        return signal[:, :SIGNAL_LEN]
    if T < SIGNAL_LEN:
        return np.pad(signal, ((0, 0), (0, SIGNAL_LEN - T)), mode='constant')
    return signal


# ── ECG augmentation (training only) ─────────────────────────────────────────

def augment_signal(signal: np.ndarray) -> np.ndarray:
    """Apply random ECG augmentations (training set only).

    1. Random time shift  — ±AUG_TIME_SHIFT samples using np.roll
    2. Amplitude scaling  — scale ∈ [1-AUG_AMPLITUDE_SCALE, 1+AUG_AMPLITUDE_SCALE]
    3. Gaussian noise     — σ ∈ [AUG_NOISE_MIN, AUG_NOISE_MAX]
    4. Lead dropout       — zero one random lead with prob AUG_LEAD_DROPOUT_PROB
    """
    signal = signal.copy()

    shift = np.random.randint(-AUG_TIME_SHIFT, AUG_TIME_SHIFT + 1)
    if shift != 0:
        signal = np.roll(signal, shift, axis=-1)

    scale  = np.random.uniform(1.0 - AUG_AMPLITUDE_SCALE, 1.0 + AUG_AMPLITUDE_SCALE)
    signal *= scale

    sigma  = np.random.uniform(AUG_NOISE_MIN, AUG_NOISE_MAX)
    signal += np.random.normal(0.0, sigma, signal.shape).astype(np.float32)

    if np.random.rand() < AUG_LEAD_DROPOUT_PROB:
        lead_idx = np.random.randint(signal.shape[0])
        signal[lead_idx] = 0.0

    return signal


# ── Dataset ───────────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """PTB-XL dataset with multi-label subclass targets.

    Labels are binary vectors of length len(CLASSES) where each position
    corresponds to a specific diagnostic subclass (e.g. IMI, CLBBB, LVH)
    rather than a superclass (MI, CD, HYP).
    """

    def __init__(self, df: pd.DataFrame, data_dir: str = DATA_DIR, augment: bool = False):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row         = self.df.iloc[idx]
        rel_path    = row['filename_hr'].replace('.hea', '')
        record_path = os.path.join(self.data_dir, rel_path)
        record      = wfdb.rdrecord(record_path)
        signal      = record.p_signal.T.astype(np.float32)

        signal = pad_or_crop(signal)
        signal = bandpass_filter(signal)
        signal = instance_normalize(signal)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            signal = augment_signal(signal)

        label = row[CLASSES].values.astype(np.float32)
        return torch.from_numpy(signal), torch.from_numpy(label)


# ── PTB-XL metadata loader ────────────────────────────────────────────────────

def load_ptbxl(data_dir: str = DATA_DIR) -> tuple:
    """Load PTB-XL metadata and assign per-subclass binary labels.

    KEY CHANGE vs v1:
    The original code mapped SCP codes → 5 diagnostic superclasses.
    This version labels records directly with the SCP subclass codes
    (e.g. 'IMI', 'CLBBB', 'LVH') that are present in config.CLASSES.

    Only codes with annotator confidence = 100.0 are accepted.
    Records with no matching subclass are dropped.
    """
    csv_path = os.path.join(data_dir, 'ptbxl_database.csv')

    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Build a set for O(1) membership checks
    classes_set = set(CLASSES)

    def parse_scp(raw: str) -> list[str]:
        """Return list of CLASSES codes present in this record at confidence=100."""
        if not isinstance(raw, str):
            return []
        try:
            codes = ast.literal_eval(raw)
        except Exception:
            return []
        return [
            code for code, confidence in codes.items()
            if confidence >= 100.0 and code in classes_set
        ]

    df['subclass_list'] = df['scp_codes'].apply(parse_scp)

    # One binary column per subclass
    for cls in CLASSES:
        df[cls] = df['subclass_list'].apply(lambda x: 1.0 if cls in x else 0.0)

    # Drop records that matched none of our subclasses
    df = df[df[CLASSES].sum(axis=1) > 0].copy()

    # Report label statistics so you can spot severely under-represented classes
    print("Subclass label counts (records with confidence=100 annotation):")
    counts = {cls: int(df[cls].sum()) for cls in CLASSES}
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * min(cnt // 100, 40)
        print(f"  {cls:<8} {cnt:>5}  {bar}")
    print()

    test_df  = df[df['strat_fold'] == TEST_FOLD]
    val_df   = df[df['strat_fold'] == VAL_FOLD]
    train_df = df[~df['strat_fold'].isin([TEST_FOLD, VAL_FOLD])]

    print(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(data_dir:    str = DATA_DIR,
                    batch_size:  int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS) -> tuple:
    """Return (train_loader, val_loader, test_loader).

    Augmentation is enabled on the train loader only.
    """
    train_df, val_df, test_df = load_ptbxl(data_dir)

    kwargs = dict(pin_memory=True, persistent_workers=(num_workers > 0),
                  num_workers=num_workers)

    train_loader = DataLoader(
        PTBXLDataset(train_df, data_dir, augment=True),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        PTBXLDataset(val_df, data_dir, augment=False),
        batch_size=batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        PTBXLDataset(test_df, data_dir, augment=False),
        batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader, test_loader