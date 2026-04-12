"""
train.py — Training loop for ECGAttentionNet (improved v2).

Key changes vs v1:
  1. CLASS-WEIGHTED FOCAL LOSS
       Replaces plain BCEWithLogitsLoss.
       • Pos-weight per class is computed from the training set label frequency,
         so rare classes (MI, HYP) get amplified proportionally to their
         under-representation — the single biggest fix for minority-class AUC.
       • Focal modulation (gamma=2) further down-weights easy negatives so the
         model is forced to learn harder examples.
       Both can be disabled independently via FOCAL_GAMMA=0 / USE_CLASS_WEIGHTS=False
       in config.py.

  2. PER-CLASS THRESHOLD OPTIMISATION
       After each epoch, the threshold that maximises F1 on the *validation set*
       is searched independently for each class over [0.1, 0.9].
       Best thresholds are logged and saved into the checkpoint.
       At inference time, load thresholds from the checkpoint instead of using 0.5.

  3. LABEL SMOOTHING
       Labels are soft-clipped to [eps, 1-eps] with LABEL_SMOOTHING=0.05.
       Reduces overconfidence and improves calibration.

  4. COSINE ANNEALING WITH WARM RESTARTS (optional)
       Set LR_SCHEDULER='cosine_restarts' in config.py to switch from OneCycleLR.
       Useful for longer runs (>50 epochs) where OneCycleLR can prematurely
       collapse the LR.

  5. EARLY STOPPING
       Stops if val macro-AUC does not improve for EARLY_STOPPING_PATIENCE epochs.
       Set EARLY_STOPPING_PATIENCE=0 to disable.

All defaults still come from config.py.
Run:
  python train.py
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_WARMUP_PCT,
    GRAD_CLIP_NORM,
    NUM_WORKERS,
    NUM_CLASSES,
    BASE_CH,
    N_HEADS,
    DROPOUT,
)
from dataset import get_dataloaders
from model   import ECGAttentionNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Optional config additions (safe defaults if not present in config.py) ──────

def _cfg(name, default):
    try:
        import config as _c
        return getattr(_c, name)
    except AttributeError:
        return default

FOCAL_GAMMA            = _cfg('FOCAL_GAMMA',            2.0)    # 0 = plain BCE
USE_CLASS_WEIGHTS      = _cfg('USE_CLASS_WEIGHTS',      True)
LABEL_SMOOTHING        = _cfg('LABEL_SMOOTHING',        0.05)
LR_SCHEDULER           = _cfg('LR_SCHEDULER',           'onecycle')  # or 'cosine_restarts'
EARLY_STOPPING_PATIENCE= _cfg('EARLY_STOPPING_PATIENCE', 10)   # 0 = disabled
COSINE_T0              = _cfg('COSINE_T0',              10)     # epochs per restart


# ── Focal loss ─────────────────────────────────────────────────────────────────

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Binary focal loss (per-label, multi-label safe).

    FL(p) = -α · (1-p)^γ · log(p)   for positives
    FL(p) = -(1-α) · p^γ · log(1-p)  for negatives

    pos_weight mirrors the BCEWithLogitsLoss convention:
        pos_weight[c] = (#neg_c / #pos_c)  — applied as α_c.

    gamma=0 reduces to weighted BCE (no focal modulation).
    """

    def __init__(self,
                 pos_weight:     torch.Tensor | None = None,
                 gamma:          float               = 2.0,
                 label_smoothing: float              = 0.0):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Standard BCE per element
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        if self.gamma == 0:
            return bce.mean()

        # Focal modulation
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def compute_pos_weights(train_loader, num_classes: int, device) -> torch.Tensor:
    """
    Compute pos_weight = (#neg / #pos) per class from training labels.
    Clamped to [1, 50] to prevent extreme values.
    """
    print('  Computing class frequencies for pos_weight...')
    pos_counts = torch.zeros(num_classes)
    total      = 0

    for _, labels in train_loader:
        pos_counts += labels.sum(dim=0)
        total      += labels.shape[0]

    neg_counts  = total - pos_counts
    pos_weights = (neg_counts / pos_counts.clamp(min=1)).clamp(1.0, 50.0)

    for i, cls in enumerate(CLASSES):
        print(f'    {cls}: pos={int(pos_counts[i])}, neg={int(neg_counts[i])}, '
              f'pos_weight={pos_weights[i]:.2f}')

    return pos_weights.to(device)


# ── Per-class threshold search ─────────────────────────────────────────────────

def find_best_thresholds(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         search_values: np.ndarray = np.arange(0.1, 0.91, 0.02)
                         ) -> np.ndarray:
    """
    For each class, search `search_values` for the threshold that maximises F1.
    Returns array of shape (num_classes,).
    """
    best_thresh = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        best_f1 = -1.0
        for t in search_values:
            f1 = f1_score(y_true[:, c], (y_pred[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh[c] = f1, float(t)
    return best_thresh


# ── Device info ────────────────────────────────────────────────────────────────

def print_device_info():
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA   : {torch.version.cuda}")


def get_vram_str() -> str:
    if DEVICE.type != 'cuda':
        return 'N/A'
    used  = torch.cuda.memory_allocated()  / 1e6
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    return f"{used:.0f}/{total:.0f} MB"


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_aucs(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    aucs = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            aucs[cls] = roc_auc_score(y_true[:, i], y_pred[:, i])
    aucs['macro'] = float(np.mean(list(aucs.values())))
    return aucs


def compute_f1s(y_true: np.ndarray,
                y_pred_prob: np.ndarray,
                thresholds: np.ndarray | None = None) -> dict:
    """F1 per class. Uses per-class thresholds if provided, else 0.5."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    y_bin = (y_pred_prob >= thresholds).astype(int)
    f1s = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            f1s[cls] = f1_score(y_true[:, i], y_bin[:, i], zero_division=0)
    f1s['macro'] = float(np.mean(list(f1s.values())))
    return f1s


def compute_precision_recall(y_true: np.ndarray,
                              y_pred_prob: np.ndarray,
                              thresholds: np.ndarray | None = None) -> tuple[dict, dict]:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    y_bin = (y_pred_prob >= thresholds).astype(int)
    prec, rec = {}, {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            prec[cls] = precision_score(y_true[:, i], y_bin[:, i], zero_division=0)
            rec[cls]  = recall_score(   y_true[:, i], y_bin[:, i], zero_division=0)
    prec['macro'] = float(np.mean(list(prec.values())))
    rec['macro']  = float(np.mean(list(rec.values())))
    return prec, rec


def trend_arrow(current: float, previous: float | None, higher_is_better: bool = True) -> str:
    if previous is None:
        return ' '
    delta = current - previous
    if abs(delta) < 1e-4:
        return '→'
    improving = (delta > 0) if higher_is_better else (delta < 0)
    return '↑' if improving else '↓'


def loss_trend(current, previous):
    return trend_arrow(current, previous, higher_is_better=False)

def auc_trend(current, previous):
    return trend_arrow(current, previous, higher_is_better=True)


# ── Train / eval passes ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion,
              optimizer=None, scaler=None, scheduler=None,
              training: bool = False,
              scheduler_step_per_batch: bool = True,
              ) -> tuple[float, np.ndarray, np.ndarray]:
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for signals, labels in loader:
            signals = signals.to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels  = labels.to(DEVICE,  dtype=torch.float32, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type):
                logits = model(signals)
                loss   = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        np.vstack(all_labels),
        np.vstack(all_preds),
    )


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_epoch_header():
    print()
    print(f"{'Ep':>4}  {'T-Loss':>7}  {'V-Loss':>7}  "
          f"{'T-AUC':>6}  {'V-AUC':>6}  {'V-F1':>6}  "
          f"{'LR':>8}  {'VRAM':>12}  {'Time':>6}")
    print("─" * 90)


def print_epoch_row(epoch, epochs,
                    t_loss, v_loss,
                    t_aucs, v_aucs, v_f1s,
                    lr, vram, elapsed,
                    prev_v_loss=None, prev_v_auc=None):
    l_arrow   = loss_trend(v_loss,         prev_v_loss)
    auc_arrow = auc_trend(v_aucs['macro'], prev_v_auc)
    print(
        f"{epoch:>3}/{epochs:<3}  "
        f"{t_loss:>7.4f}  {v_loss:>6.4f}{l_arrow}  "
        f"{t_aucs['macro']:>6.3f}  {v_aucs['macro']:>5.3f}{auc_arrow}  "
        f"{v_f1s['macro']:>6.3f}  "
        f"{lr:>8.2e}  {vram:>12}  {elapsed:>5.1f}s"
    )


def print_per_class(v_aucs, v_f1s, v_prec, v_rec, thresholds=None):
    print()
    print(f"  {'Class':<6}  {'AUC':>6}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}"
          + (f"  {'Thresh':>6}" if thresholds is not None else ''))
    print("  " + "─" * (38 + (9 if thresholds is not None else 0)))
    for i, cls in enumerate(CLASSES):
        auc = v_aucs.get(cls, float('nan'))
        f1  = v_f1s.get(cls,  float('nan'))
        p   = v_prec.get(cls, float('nan'))
        r   = v_rec.get(cls,  float('nan'))
        t_  = f"  {thresholds[i]:>6.2f}" if thresholds is not None else ''
        print(f"  {cls:<6}  {auc:>6.3f}  {f1:>6.3f}  {p:>6.3f}  {r:>6.3f}{t_}")
    print(f"  {'macro':<6}  {v_aucs['macro']:>6.3f}  {v_f1s['macro']:>6.3f}  "
          f"{v_prec['macro']:>6.3f}  {v_rec['macro']:>6.3f}")


def print_summary(history: list[dict], best_epoch: int):
    print()
    print("═" * 90)
    print("TRAINING COMPLETE — SUMMARY")
    print("═" * 90)
    best = history[best_epoch - 1]
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Best val AUC    : {best['v_auc']:.4f}")
    print(f"  Best val F1     : {best['v_f1']:.4f}")
    print(f"  Val loss at best: {best['v_loss']:.4f}")
    print()
    print("  Epoch-by-epoch val macro-AUC:")
    bar_max = max(h['v_auc'] for h in history)
    for h in history:
        filled = int((h['v_auc'] / bar_max) * 30)
        marker = '★' if h['epoch'] == best_epoch else ' '
        print(f"  Ep {h['epoch']:>2} {marker} {'█' * filled}{'░' * (30 - filled)}  {h['v_auc']:.3f}")
    print("═" * 90)


# ── Main training loop ─────────────────────────────────────────────────────────

def train(data_dir:       str   = DATA_DIR,
          epochs:         int   = EPOCHS,
          batch_size:     int   = BATCH_SIZE,
          lr:             float = LEARNING_RATE,
          checkpoint_dir: str   = CHECKPOINT_DIR,
          num_workers:    int   = NUM_WORKERS):

    print_device_info()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print('\nLoading PTB-XL...')
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        num_workers = num_workers,
    )

    # ── Loss function ──────────────────────────────────────────────────────────
    pos_weight = compute_pos_weights(train_loader, NUM_CLASSES, DEVICE) if USE_CLASS_WEIGHTS else None
    criterion  = FocalBCEWithLogitsLoss(
        pos_weight      = pos_weight,
        gamma           = FOCAL_GAMMA,
        label_smoothing = LABEL_SMOOTHING,
    ).to(DEVICE)
    print(f'\nLoss: FocalBCE  gamma={FOCAL_GAMMA}  label_smoothing={LABEL_SMOOTHING}')
    print(f'Class weights : {"on" if USE_CLASS_WEIGHTS else "off"}')

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ECGAttentionNet(
        num_classes = NUM_CLASSES,
        base_ch     = BASE_CH,
        nhead       = N_HEADS,
        dropout     = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)

    if LR_SCHEDULER == 'cosine_restarts':
        # Step per epoch, not per batch
        scheduler             = CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_T0)
        scheduler_per_batch   = False
        print(f'Scheduler: CosineAnnealingWarmRestarts  T_0={COSINE_T0}')
    else:
        scheduler           = OneCycleLR(
            optimizer,
            max_lr          = lr,
            total_steps     = total_steps,
            pct_start       = LR_WARMUP_PCT,
            anneal_strategy = 'cos',
        )
        scheduler_per_batch = True
        print(f'Scheduler: OneCycleLR  warmup={LR_WARMUP_PCT*100:.0f}%')

    scaler    = GradScaler(device=DEVICE.type)
    save_path = os.path.join(checkpoint_dir, os.path.basename(CHECKPOINT_PATH))

    print(f'\nTraining {epochs} epochs  batch={batch_size}  lr={lr}')
    if EARLY_STOPPING_PATIENCE > 0:
        print(f'Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs')
    print_epoch_header()

    best_val_auc     = 0.0
    best_epoch       = 1
    history          = []
    prev_v_loss      = None
    prev_v_auc       = None
    no_improve_count = 0
    best_thresholds  = np.full(NUM_CLASSES, 0.5)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train pass ─────────────────────────────────────────────────────────
        t_loss, t_labels, t_preds = run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer, scaler=scaler,
            scheduler=scheduler if scheduler_per_batch else None,
            training=True,
            scheduler_step_per_batch=scheduler_per_batch,
        )
        if not scheduler_per_batch:
            scheduler.step()

        # ── Val pass ───────────────────────────────────────────────────────────
        v_loss, v_labels, v_preds = run_epoch(
            model, val_loader, criterion, training=False
        )

        elapsed = time.time() - t0

        # ── Per-class threshold search (on val set) ────────────────────────────
        best_thresholds = find_best_thresholds(v_labels, v_preds)

        # ── Metrics ────────────────────────────────────────────────────────────
        t_aucs        = compute_aucs(t_labels, t_preds)
        v_aucs        = compute_aucs(v_labels, v_preds)
        v_f1s         = compute_f1s(v_labels, v_preds, best_thresholds)
        v_prec, v_rec = compute_precision_recall(v_labels, v_preds, best_thresholds)
        current_lr    = optimizer.param_groups[0]['lr']
        vram          = get_vram_str()

        print_epoch_row(
            epoch, epochs,
            t_loss, v_loss,
            t_aucs, v_aucs, v_f1s,
            current_lr, vram, elapsed,
            prev_v_loss, prev_v_auc
        )

        if epoch % 5 == 0 or epoch == epochs:
            print_per_class(v_aucs, v_f1s, v_prec, v_rec, best_thresholds)
            print()

        # ── Save best checkpoint ───────────────────────────────────────────────
        if v_aucs['macro'] > best_val_auc:
            best_val_auc     = v_aucs['macro']
            best_epoch       = epoch
            no_improve_count = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_auc':          best_val_auc,
                'val_aucs':         v_aucs,
                'val_f1s':          v_f1s,
                # ── NEW: save per-class thresholds with the checkpoint ──────────
                'thresholds':       best_thresholds.tolist(),
                'config': {
                    'BASE_CH':    BASE_CH,
                    'N_HEADS':    N_HEADS,
                    'DROPOUT':    DROPOUT,
                    'BATCH_SIZE': batch_size,
                    'EPOCHS':     epochs,
                    'LR':         lr,
                    'FOCAL_GAMMA':            FOCAL_GAMMA,
                    'USE_CLASS_WEIGHTS':      USE_CLASS_WEIGHTS,
                    'LABEL_SMOOTHING':        LABEL_SMOOTHING,
                }
            }, save_path)
            print(f"  ✓ New best  macro-AUC={best_val_auc:.4f}  macro-F1={v_f1s['macro']:.4f}"
                  f"  → {save_path}")
            thresh_str = '  '.join(f'{c}:{best_thresholds[i]:.2f}' for i, c in enumerate(CLASSES))
            print(f"    Thresholds: {thresh_str}")
        else:
            no_improve_count += 1

        history.append({
            'epoch':  epoch,
            't_loss': t_loss,
            'v_loss': v_loss,
            't_auc':  t_aucs['macro'],
            'v_auc':  v_aucs['macro'],
            'v_f1':   v_f1s['macro'],
        })

        prev_v_loss = v_loss
        prev_v_auc  = v_aucs['macro']

        # ── Early stopping ─────────────────────────────────────────────────────
        if EARLY_STOPPING_PATIENCE > 0 and no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(f'\n  ⚡ Early stopping triggered — no improvement for {EARLY_STOPPING_PATIENCE} epochs.')
            break

    print_summary(history, best_epoch)
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ECGAttentionNet v2. Defaults come from config.py.')
    parser.add_argument('--data_dir',       type=str,   default=DATA_DIR)
    parser.add_argument('--epochs',         type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',     type=int,   default=BATCH_SIZE,
                        help='Lower to 8 if CUDA OOM on RTX 3050')
    parser.add_argument('--lr',             type=float, default=LEARNING_RATE)
    parser.add_argument('--checkpoint_dir', type=str,   default=CHECKPOINT_DIR)
    parser.add_argument('--num_workers',    type=int,   default=NUM_WORKERS)
    args = parser.parse_args()

    train(
        data_dir       = args.data_dir,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        checkpoint_dir = args.checkpoint_dir,
        num_workers    = args.num_workers,
    )