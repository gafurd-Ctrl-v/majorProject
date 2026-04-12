"""
model.py — ECGAttentionNet architecture (improved v2).

Key changes vs v1:
  • ResidualBlock1D now includes a Squeeze-and-Excitation (SE) channel
    attention module. SE learns which feature channels are most informative
    per lead, consistently adding ~0.5–1 AUC point at negligible cost.
  • ECGResNetBackbone gains an optional 4th residual block (layer4) so the
    receptive field covers full 5-second windows. Controlled by BASE_CH:
      BASE_CH=32 → 4-block backbone, out_channels=512 (default, recommended)
      BASE_CH=16 → lighter model if VRAM is tight
  • ECGAttentionNet now supports 2-layer stacked attention (num_attn_layers)
    for richer inter-lead interaction.
  • _init_weights extended to LayerNorm and positional encoding.

All hyperparameters still come from config.py. Suggested additions to config.py:
  SE_REDUCTION   = 16    # squeeze-and-excitation bottleneck ratio
  NUM_ATTN_LAYERS = 2    # number of stacked attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, BASE_CH, N_HEADS, DROPOUT

# Optional config additions — fall back to sensible defaults if not in config.py
try:
    from config import SE_REDUCTION
except ImportError:
    SE_REDUCTION = 16

try:
    from config import NUM_ATTN_LAYERS
except ImportError:
    NUM_ATTN_LAYERS = 2


# ── Building blocks ────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel attention: globally pool → bottleneck FC → sigmoid gates.

    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    Ratio of 16 is standard; lower (e.g. 4) for very small channel counts.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        s = self.pool(x).squeeze(-1)   # (N, C)
        s = self.fc(s).unsqueeze(-1)   # (N, C, 1)
        return x * s


class ResidualBlock1D(nn.Module):
    """1-D residual block with optional downsampling skip + SE attention."""

    def __init__(self,
                 in_ch:     int,
                 out_ch:    int,
                 stride:    int   = 1,
                 dropout:   float = DROPOUT,
                 use_se:    bool  = True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1,      padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.se    = SqueezeExcitation(out_ch) if use_se else nn.Identity()
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.skip(x))


class ECGResNetBackbone(nn.Module):
    """Shared 1-D ResNet + SE processing a single ECG lead.

    Input:  (N, 1, 5000)
    Output: (N, BASE_CH*16, T')

    Channel progression with BASE_CH=32:
      stem   → (N,  32, 1250)
      layer1 → (N,  64,  625)
      layer2 → (N, 128,  313)
      layer3 → (N, 256,  157)   ← Grad-CAM target (unchanged from v1)
      layer4 → (N, 512,   79)   ← NEW: wider receptive field
    """

    def __init__(self, base_ch: int = BASE_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResidualBlock1D(base_ch,      base_ch * 2,  stride=2)
        self.layer2 = ResidualBlock1D(base_ch * 2,  base_ch * 4,  stride=2)
        self.layer3 = ResidualBlock1D(base_ch * 4,  base_ch * 8,  stride=2)
        self.layer4 = ResidualBlock1D(base_ch * 8,  base_ch * 16, stride=2)  # NEW
        self.out_channels = base_ch * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ── Stacked attention block ────────────────────────────────────────────────────

class LeadAttentionBlock(nn.Module):
    """One layer of: MultiHeadAttention → Add&Norm → FF → Add&Norm."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_w


# ── Main model ─────────────────────────────────────────────────────────────────

class ECGAttentionNet(nn.Module):
    """
    Full ECG classification model (v2 — improved).

    Changes from v1:
      • SE blocks in each residual layer for channel-level attention.
      • 4-block backbone for a wider temporal receptive field.
      • Stacked attention (num_attn_layers=2) for richer lead interactions.
      • Attention weights from the *last* layer cached (same API as v1).
    """

    def __init__(self,
                 num_classes:     int   = NUM_CLASSES,
                 base_ch:         int   = BASE_CH,
                 nhead:           int   = N_HEADS,
                 dropout:         float = DROPOUT,
                 num_attn_layers: int   = NUM_ATTN_LAYERS):
        super().__init__()
        self.num_leads = 12
        self.backbone  = ECGResNetBackbone(base_ch)
        d_model        = self.backbone.out_channels   # BASE_CH * 16

        # Learnable positional encoding — one vector per lead
        self.lead_pos_enc = nn.Parameter(torch.zeros(1, 12, d_model))

        # Stacked attention layers
        self.attn_layers = nn.ModuleList([
            LeadAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_attn_layers)
        ])

        # Classification head — slightly deeper
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Attention weights cached after each forward pass (for explainability)
        self._attn_weights: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self):
        # Positional encoding: small normal (not zeros after init)
        nn.init.trunc_normal_(self.lead_pos_enc, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 5000)
        Returns:
            logits: (B, num_classes) — raw logits; apply sigmoid for probabilities.
        """
        B, L, T = x.shape

        # CNN per lead (shared weights)
        x    = x.reshape(B * L, 1, T)
        feat = self.backbone(x).mean(dim=-1)   # (B×12, d_model)
        feat = feat.reshape(B, L, -1)           # (B, 12, d_model)

        # Lead positional encoding
        feat = feat + self.lead_pos_enc

        # Stacked multi-head attention
        last_attn_w = None
        for layer in self.attn_layers:
            feat, last_attn_w = layer(feat)

        self._attn_weights = last_attn_w.detach().cpu()   # (B, 12, 12)

        feat = feat.mean(dim=1)   # (B, d_model)
        return self.head(feat)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Returns (B, 12, 12) attention matrix from the last forward pass."""
        return self._attn_weights