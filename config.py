import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — change these to match your system
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR        = r"C:\Users\soham\Downloads\MJPR\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHECKPOINT_DIR  = 'checkpoints'
CHECKPOINT_FILE = 'best_model.pt'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FS         = 500
SIGNAL_LEN = 5000

# ── Expanded class list: 27 diagnostic subclasses instead of 5 superclasses ──
#
# Previously the model predicted 5 coarse superclasses (NORM/MI/STTC/CD/HYP).
# Now it predicts 27 specific SCP subclasses, e.g. IMI (Inferior MI) vs ASMI
# (Anteroseptal MI) — classes the old model lumped together under "MI".
#
# Approximate PTB-XL record counts per class (confidence=100 only):
#   NORM: 9514  IMI: 1922  NDT: 1872  NST_: 1295  IRBBB: 1385
#   ASMI: 1016  LVH: 1079  CLBBB: 597  CRBBB: 544  ILMI: 769
#   AMI:  570   RVH:  435   DIG:  349  LNGQT: 289  ANEUR: 239
#   ALMI: 427   INJAS: 378  IVCD: 276  LAFB:  144  LMI:  303
#   INJAL: 230  IPLMI: 219  IPMI: 178  WPW:   148  INJIN: 148
#   PMI:  112   SEHYP: 146
#
# Classes with <100 samples (e.g. LPFB: 53) are omitted — too few for reliable
# generalisation even with focal loss and class weighting.

CLASSES = [
    # ── Normal ────────────────────────────────────────────────────────────────
    'NORM',

    # ── Myocardial Infarction subclasses ─────────────────────────────────────
    'IMI',    # Inferior MI (RCA territory)
    'ASMI',   # Anteroseptal MI (proximal LAD)
    'ILMI',   # Inferolateral MI (RCA + LCx)
    'AMI',    # Anterior MI (LAD)
    'ALMI',   # Anterolateral MI (LAD + LCx)
    'INJAS',  # Injury pattern, anteroseptal
    'LMI',    # Lateral MI (LCx territory)
    'INJAL',  # Injury pattern, anterolateral
    'IPLMI',  # Inferoposterolateral MI
    'IPMI',   # Inferoposterior MI
    'INJIN',  # Injury pattern, inferior
    'PMI',    # Posterior MI (circumflex / RCA)

    # ── ST/T Change subclasses ────────────────────────────────────────────────
    'NDT',    # Nonspecific T-wave abnormality
    'NST_',   # Nonspecific ST changes
    'DIG',    # Digitalis effect (ST depression/T-wave inversion)
    'LNGQT',  # Long QT interval
    'ANEUR',  # ST elevation — ventricular aneurysm pattern

    # ── Conduction Disturbance subclasses ─────────────────────────────────────
    'IRBBB',  # Incomplete right bundle branch block
    'CLBBB',  # Complete left bundle branch block
    'CRBBB',  # Complete right bundle branch block
    'IVCD',   # Nonspecific intraventricular conduction delay
    'LAFB',   # Left anterior fascicular block
    'WPW',    # Wolff-Parkinson-White

    # ── Hypertrophy subclasses ────────────────────────────────────────────────
    'LVH',    # Left ventricular hypertrophy
    'RVH',    # Right ventricular hypertrophy
    'SEHYP',  # Subendocardial hypertrophy / strain pattern
]

# Maps each subclass back to its diagnostic superclass.
# Used by diagnosis_engine.py to group related findings.
CLASS_SUPERCLASS = {
    'NORM':  'NORM',
    'IMI':   'MI',   'ASMI': 'MI',  'ILMI':  'MI',  'AMI':   'MI',
    'ALMI':  'MI',   'INJAS': 'MI', 'LMI':   'MI',  'INJAL': 'MI',
    'IPLMI': 'MI',   'IPMI': 'MI',  'INJIN': 'MI',  'PMI':   'MI',
    'NDT':   'STTC', 'NST_': 'STTC','DIG':   'STTC','LNGQT': 'STTC',
    'ANEUR': 'STTC',
    'IRBBB': 'CD',   'CLBBB': 'CD', 'CRBBB': 'CD',  'IVCD':  'CD',
    'LAFB':  'CD',   'WPW':   'CD',
    'LVH':   'HYP',  'RVH':  'HYP', 'SEHYP': 'HYP',
}

# Human-readable names for all 27 subclasses
CLASS_NAMES = {
    'NORM':  'Normal Sinus Rhythm',
    'IMI':   'Inferior MI',
    'ASMI':  'Anteroseptal MI',
    'ILMI':  'Inferolateral MI',
    'AMI':   'Anterior MI',
    'ALMI':  'Anterolateral MI',
    'INJAS': 'Injury — Anteroseptal',
    'LMI':   'Lateral MI',
    'INJAL': 'Injury — Anterolateral',
    'IPLMI': 'Inferoposterolateral MI',
    'IPMI':  'Inferoposterior MI',
    'INJIN': 'Injury — Inferior',
    'PMI':   'Posterior MI',
    'NDT':   'T-Wave Abnormality',
    'NST_':  'ST Changes',
    'DIG':   'Digitalis Effect',
    'LNGQT': 'Long QT Interval',
    'ANEUR': 'Ventricular Aneurysm',
    'IRBBB': 'Incomplete RBBB',
    'CLBBB': 'Complete LBBB',
    'CRBBB': 'Complete RBBB',
    'IVCD':  'Intraventricular Delay',
    'LAFB':  'Left Anterior Fascicular Block',
    'WPW':   'Wolff-Parkinson-White',
    'LVH':   'Left Ventricular Hypertrophy',
    'RVH':   'Right Ventricular Hypertrophy',
    'SEHYP': 'Subendocardial Hypertrophy',
}

# Convenience sets — used by diagnosis_engine.py for lead-territory mapping
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3',  'V4',  'V5',  'V6']

LEAD_II_INDEX = 1

TEST_FOLD = 10
VAL_FOLD  =  9

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

FILTER_LOWCUT  =  0.5
FILTER_HIGHCUT = 45.0
FILTER_ORDER   =  4

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES     = len(CLASSES)   # 27
BASE_CH         = 32
N_HEADS         = 4
DROPOUT         = 0.2
SE_REDUCTION    = 16
NUM_ATTN_LAYERS = 2

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE      = 64
EPOCHS          = 60    # increased from 49 — more classes need more training time
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
LR_WARMUP_PCT   = 0.1
GRAD_CLIP_NORM  = 1.0
NUM_WORKERS     = 0

# Focal loss — critical for the highly imbalanced subclass distribution.
# With 27 classes and minority classes having only ~100-200 samples vs
# NORM's 9500+, gamma=2 + USE_CLASS_WEIGHTS=True is strongly recommended.
FOCAL_GAMMA     = 2.0
USE_CLASS_WEIGHTS = True

# Label smoothing — reduces overconfidence, especially important with small
# minority classes where the model may never see hard negatives.
LABEL_SMOOTHING = 0.05

LR_SCHEDULER    = 'onecycle'
COSINE_T0       = 10

# Increased patience for 27 classes — minority classes often improve late
EARLY_STOPPING_PATIENCE = 12

# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

AUG_TIME_SHIFT        = 200
AUG_AMPLITUDE_SCALE   = 0.25
AUG_NOISE_MIN         = 0.005
AUG_NOISE_MAX         = 0.02
AUG_LEAD_DROPOUT_PROB = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

TOP_K_LEADS = 3