import os
import json
import math
import copy
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


class FeatureDataset(Dataset):
    """PyTorch dataset wrapping windowed EEG features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_windowed_features(features_dir: str) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[str]]:
    """Load windowed feature tensors along with meta information."""
    data_list: List[np.ndarray] = []
    labels: List[int] = []
    file_info: List[Dict] = []
    feature_names: List[str] = []

    feature_files = [f for f in os.listdir(features_dir) if f.endswith("_windowed_features.csv")]
    if not feature_files:
        raise FileNotFoundError(f"No windowed feature files found in {features_dir}")

    for filename in feature_files:
        file_path = os.path.join(features_dir, filename)
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        id_cols = [
            "filename",
            "subject_id",
            "condition",
            "trial_id",
            "channel",
            "window_index",
            "start_time",
            "end_time",
        ]
        current_features = [col for col in df.columns if col not in id_cols]
        if not feature_names:
            feature_names = current_features
        elif feature_names != current_features:
            raise ValueError(f"Feature columns mismatch in file {filename}")

        channels = df["channel"].unique()
        windows = sorted(df["window_index"].unique())
        sample = np.zeros((len(windows), len(channels), len(current_features)), dtype=np.float32)
        for w_idx, window in enumerate(windows):
            for c_idx, channel in enumerate(channels):
                mask = (df["window_index"] == window) & (df["channel"] == channel)
                if mask.any():
                    sample[w_idx, c_idx, :] = df.loc[mask, current_features].values[0]

        label = 1 if "Afterdis" in filename else 0
        data_list.append(sample)
        labels.append(label)
        file_info.append(
            {
                "filename": filename,
                "subject_id": str(df["subject_id"].iloc[0]),
                "condition": str(df["condition"].iloc[0]),
                "trial_id": str(df["trial_id"].iloc[0]),
                "label": int(label),
            }
        )

    if not data_list:
        raise ValueError("No feature tensors could be loaded")

    data_array = np.stack(data_list, axis=0)
    labels_array = np.array(labels)
    return data_array, labels_array, file_info, feature_names


def subject_group_standardize(data: np.ndarray, subject_ids: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-subject standardization to mitigate inter-subject scale drift."""
    standardized = np.zeros_like(data, dtype=np.float32)
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        subject_data = data[mask].astype(np.float32)
        flattened = subject_data.reshape(-1, subject_data.shape[-1])
        mean = flattened.mean(axis=0, keepdims=True)
        std = flattened.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        normalized = (flattened - mean) / std
        standardized[mask] = normalized.reshape(subject_data.shape)
    return standardized


# -----------------------------------------------------------------------------
# Model zoo definitions
# -----------------------------------------------------------------------------


class TemporalConvNet(nn.Module):
    """Stacked dilated causal convolutions as used in TCN."""

    def __init__(self, input_channels: int, channel_sizes: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(channel_sizes)
        for i in range(num_levels):
            in_channels = input_channels if i == 0 else channel_sizes[i - 1]
            out_channels = channel_sizes[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout(dropout),
                )
            )
        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.network:
            residual = out
            out = block(out)
            out = out[..., : residual.size(-1)]  # remove padding tail
            if residual.size(1) != out.size(1):
                residual = nn.functional.pad(residual, (0, 0, 0, out.size(1) - residual.size(1)))
            out = out + residual
        return out


class FeatureTCNClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2):
        super().__init__()
        embed_dim = 128
        self.project = nn.Sequential(
            nn.Conv1d(num_channels * num_features, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.tcn = TemporalConvNet(embed_dim, [128, 128, 128], kernel_size=3, dropout=0.3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features).permute(0, 2, 1)
        x = self.project(x)
        x = self.tcn(x)
        return self.head(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], bottleneck: int = 32, dropout: float = 0.2):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1) if in_channels > bottleneck else nn.Identity()
        branches = []
        for k in kernel_sizes:
            branches.append(
                nn.Sequential(
                    nn.Conv1d(bottleneck if in_channels > bottleneck else in_channels, out_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )
        branches.append(nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1), nn.Conv1d(in_channels, out_channels, kernel_size=1)))
        self.branches = nn.ModuleList(branches)
        self.batch_norm = nn.BatchNorm1d(out_channels * len(branches))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(self.bottleneck(x)) if hasattr(self.bottleneck, "weight") else branch(x) for branch in self.branches[:-1]]
        outputs.append(self.branches[-1](x))
        x = torch.cat(outputs, dim=1)
        x = self.batch_norm(x)
        return self.dropout(nn.functional.gelu(x))


class InceptionTimeClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2):
        super().__init__()
        embed_dim = 128
        self.project = nn.Sequential(
            nn.Conv1d(num_channels * num_features, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.blocks = nn.Sequential(
            InceptionBlock(embed_dim, 32, [3, 5, 7], dropout=0.3),
            InceptionBlock(32 * 4, 32, [3, 5, 7], dropout=0.3),
            InceptionBlock(32 * 4, 32, [3, 5, 7], dropout=0.3),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features).permute(0, 2, 1)
        x = self.project(x)
        x = self.blocks(x)
        return self.head(x)


class TransformerTimeClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_channels * num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, time_steps, embed_dim) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.projection(x)
        if self.pos_embedding.size(1) != time_steps:
            pos = nn.functional.interpolate(self.pos_embedding.permute(0, 2, 1), size=time_steps, mode="linear", align_corners=False).permute(0, 2, 1)
        else:
            pos = self.pos_embedding
        x = self.encoder(x + pos)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class FeatureBiLSTMClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2, embed_dim: int = 128, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_channels * num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.projection(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.head(context)


class FeatureGRUClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2, embed_dim: int = 128, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_channels * num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.projection(x)
        gru_out, _ = self.gru(x)
        pooled = gru_out.mean(dim=1)
        return self.head(pooled)


class ResidualTemporalCNN(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2):
        super().__init__()
        embed_dim = 128
        self.project = nn.Sequential(
            nn.Conv1d(num_channels * num_features, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(embed_dim),
            ),
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
            ),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features).permute(0, 2, 1)
        x = self.project(x)
        residual = x
        for block in self.blocks:
            out = block(x)
            x = nn.functional.gelu(out + residual)
            residual = x
        x = self.dropout(x)
        return self.head(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, channels: int, token_dim: int, channel_dim: int):
        super().__init__()
        self.norm_token = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, tokens),
        )
        self.norm_channel = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm_token(x)
        x = x + self.token_mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mlp(self.norm_channel(x))
        return x


class MLPMixerTimeClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2):
        super().__init__()
        embed_dim = 128
        self.projection = nn.Sequential(
            nn.Linear(num_channels * num_features, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.blocks = nn.Sequential(
            MixerBlock(time_steps, embed_dim, token_dim=max(4, time_steps // 2), channel_dim=256),
            MixerBlock(time_steps, embed_dim, token_dim=max(4, time_steps // 2), channel_dim=256),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.projection(x)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))


class PatchTSTClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2, patch_size: int = 4, stride: int = 4, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.time_steps = time_steps
        self.flatten_dim = num_channels * num_features * patch_size
        self.projection = nn.Linear(self.flatten_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, dim = x.shape
        if time_steps < self.patch_size:
            pad = self.patch_size - time_steps
            x = nn.functional.pad(x, (0, 0, 0, pad))
            time_steps = x.size(1)
        patches = []
        for start in range(0, max(time_steps - self.patch_size + 1, 1), self.stride):
            end = start + self.patch_size
            if end > time_steps:
                segment = x[:, -self.patch_size :, :]
            else:
                segment = x[:, start:end, :]
            patches.append(segment.reshape(batch, -1))
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        patches = self._extract_patches(x)
        tokens = self.projection(patches)
        pos = self.pos_embedding[:, : tokens.size(1), :]
        tokens = self.encoder(tokens + pos)
        return self.head(tokens.mean(dim=1))


class TemporalFusionTransformerClassifier(nn.Module):
    def __init__(self, num_channels: int, num_features: int, time_steps: int, num_classes: int = 2, embed_dim: int = 128, hidden_size: int = 128):
        super().__init__()
        self.time_steps = time_steps
        self.input_proj = nn.Sequential(
            nn.Linear(num_channels * num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, dropout=0.2, batch_first=True)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.input_proj(x)

        enc_out, _ = self.encoder_lstm(x)
        dec_out, _ = self.decoder_lstm(x)

        context, _ = self.attention(dec_out, enc_out, enc_out)
        gate = self.fusion_gate(torch.cat([context, dec_out], dim=-1))
        fused = gate * context + (1 - gate) * dec_out
        pooled = fused.mean(dim=1)
        return self.output_layer(pooled)


# -----------------------------------------------------------------------------
# Additional modern baselines (2024-2025 popular)
# -----------------------------------------------------------------------------


class TimesBlock(nn.Module):
    """A lightweight TimesNet-style block.

    Key idea: estimate dominant periods via FFT, reshape the sequence into a
    (segments, period) 2D layout, apply 2D convolutions, then fold back.

    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(self, d_model: int, top_k: int = 3, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

        self.conv2d = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _topk_periods(x: torch.Tensor, top_k: int) -> List[int]:
        # x: (B, T, D)
        _, time_steps, _ = x.shape
        if time_steps <= 2:
            return [1]

        # FFT over time, average magnitudes across batch and channels.
        spec = torch.fft.rfft(x.float(), dim=1)  # (B, Freq, D)
        amp = spec.abs().mean(dim=(0, 2))  # (Freq,)
        amp[0] = 0.0
        top_k = max(1, min(int(top_k), int(amp.numel() - 1)))
        _, idx = torch.topk(amp, k=top_k, largest=True)

        periods: List[int] = []
        for k in idx.tolist():
            if k <= 0:
                continue
            # freq index k corresponds to ~k cycles per T samples => period ~ T/k
            p = max(1, int(round(time_steps / k)))
            periods.append(p)
        if not periods:
            periods = [1]

        # De-duplicate while keeping order.
        dedup: List[int] = []
        seen = set()
        for p in periods:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        return dedup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, time_steps, d_model = x.shape

        periods = self._topk_periods(x, self.top_k)
        outputs: List[torch.Tensor] = []

        for p in periods:
            p = max(1, min(int(p), int(time_steps)))
            pad_len = (p - (time_steps % p)) % p
            if pad_len:
                x_pad = nn.functional.pad(x, (0, 0, 0, pad_len))
            else:
                x_pad = x

            t_pad = x_pad.size(1)
            seg = t_pad // p
            # (B, T, D) -> (B, D, seg, p)
            x_2d = x_pad.view(batch, seg, p, d_model).permute(0, 3, 1, 2)
            y_2d = self.conv2d(x_2d)
            # back to (B, T, D)
            y = y_2d.permute(0, 2, 3, 1).contiguous().view(batch, t_pad, d_model)
            y = y[:, :time_steps, :]
            outputs.append(y)

        if len(outputs) == 1:
            out = outputs[0]
        else:
            out = torch.stack(outputs, dim=0).mean(dim=0)

        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class TimesNetClassifier(nn.Module):
    """TimesNet-style classifier for window-sequence features.

    Input x: (B, T, C, F)
    """

    def __init__(
        self,
        num_channels: int,
        num_features: int,
        time_steps: int,
        num_classes: int = 2,
        d_model: int = 128,
        depth: int = 2,
        top_k: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(num_channels * num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([TimesBlock(d_model=d_model, top_k=top_k, dropout=0.2) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class SimpleSSMBlock(nn.Module):
    """A compact SSM/Mamba-like block using a learnable diagonal first-order SSM.

    This is a practical, dependency-free baseline in the "SSM family":
    h_t = a * h_{t-1} + b * u_t
    y_t = c * h_t + d * u_t
    with gating + depthwise conv for local mixing.

    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(self, d_model: int, conv_kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model)

        # Stable parameterization for a in (0,1)
        self.logit_a = nn.Parameter(torch.zeros(d_model))
        self.b = nn.Parameter(torch.ones(d_model) * 0.1)
        self.c = nn.Parameter(torch.ones(d_model))
        self.d = nn.Parameter(torch.zeros(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, time_steps, d_model = x.shape

        uv = self.in_proj(x)
        u, v = torch.chunk(uv, 2, dim=-1)
        v = nn.functional.silu(v)

        # Local mixing (depthwise conv over time)
        v_local = self.dwconv(v.transpose(1, 2)).transpose(1, 2)

        a = torch.sigmoid(self.logit_a).to(v_local.dtype)  # (D,)
        b = self.b.to(v_local.dtype)
        c = self.c.to(v_local.dtype)
        d = self.d.to(v_local.dtype)

        state = torch.zeros(batch, d_model, device=v_local.device, dtype=v_local.dtype)
        ys: List[torch.Tensor] = []
        for t in range(time_steps):
            inp = v_local[:, t, :]  # (B, D)
            state = a * state + b * inp
            y = c * state + d * inp
            ys.append(y)
        y_seq = torch.stack(ys, dim=1)  # (B, T, D)

        out = u * nn.functional.silu(y_seq)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class MambaSSMClassifier(nn.Module):
    """Mamba/SSM-family baseline classifier (dependency-free).

    Input x: (B, T, C, F)
    """

    def __init__(
        self,
        num_channels: int,
        num_features: int,
        time_steps: int,
        num_classes: int = 2,
        d_model: int = 128,
        depth: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.proj = nn.Sequential(
            nn.Linear(num_channels * num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([SimpleSSMBlock(d_model=d_model, conv_kernel=3, dropout=0.2) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class TimeMixerBlock(nn.Module):
    """TimeMixer-style block: multi-scale temporal mixing + feature mixing.

    Temporal mixing is implemented with depthwise Conv1d over time using multiple
    kernel sizes, then fused. Feature mixing is an FFN over the model dimension.

    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        temporal_kernels: List[int] = [3, 5, 7],
        ffn_ratio: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=k,
                    padding=k // 2,
                    groups=d_model,
                )
                for k in temporal_kernels
            ]
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_dropout = nn.Dropout(dropout)

        hidden = int(d_model * ffn_ratio)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x_t = x.transpose(1, 2)  # (B, D, T)

        mixed = None
        for conv in self.temporal_convs:
            y = conv(x_t)
            mixed = y if mixed is None else (mixed + y)
        mixed = mixed / float(len(self.temporal_convs))
        mixed = mixed.transpose(1, 2)  # (B, T, D)
        mixed = nn.functional.gelu(mixed)
        mixed = self.temporal_dropout(mixed)
        x = self.temporal_norm(residual + mixed)

        # Feature mixing
        residual = x
        y = self.ffn(self.ffn_norm(x))
        y = self.ffn_dropout(y)
        x = residual + y
        return x


class TimeMixerClassifier(nn.Module):
    """TimeMixer baseline classifier.

    Input x: (B, T, C, F)
    """

    def __init__(
        self,
        num_channels: int,
        num_features: int,
        time_steps: int,
        num_classes: int = 2,
        d_model: int = 128,
        depth: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.proj = nn.Sequential(
            nn.Linear(num_channels * num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [TimeMixerBlock(d_model=d_model, temporal_kernels=[3, 5, 7], ffn_ratio=4, dropout=0.2) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, features = x.shape
        x = x.view(batch, time_steps, channels * features)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


MODEL_REGISTRY = {
    "tcn": FeatureTCNClassifier,
    "inception_time": InceptionTimeClassifier,
    "transformer": TransformerTimeClassifier,
    "bilstm": FeatureBiLSTMClassifier,
    "gru": FeatureGRUClassifier,
    "res_cnn": ResidualTemporalCNN,
    "mlp_mixer": MLPMixerTimeClassifier,
    "patch_tst": PatchTSTClassifier,
    "tft": TemporalFusionTransformerClassifier,
    "timesnet": TimesNetClassifier,
    "mamba_ssm": MambaSSMClassifier,
    "timemixer": TimeMixerClassifier,
}

#python Comparison/run_time_series_model_comparison.py --model tcn
# -----------------------------------------------------------------------------
# Training and evaluation helpers
# -----------------------------------------------------------------------------


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    probabilities: List[np.ndarray] = []
    predictions: List[int] = []
    true_labels: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item()
            probabilities.extend(probs.cpu().numpy())
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_arr = np.array(true_labels)
    pred_arr = np.array(predictions)
    return (
        total_loss / max(len(loader), 1),
        np.array(probabilities),
        pred_arr,
        true_arr,
        100.0 * (pred_arr == true_arr).mean(),
    )


def compute_binary_metrics(true_labels: np.ndarray, pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard binary metrics."""
    pred_classes = (pred_probs >= threshold).astype(int)
    accuracy = accuracy_score(true_labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_classes, average="binary", zero_division=0)
    auc_value = roc_auc_score(true_labels, pred_probs)
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    tn = np.logical_and(pred_classes == 0, true_labels == 0).sum()
    fp = np.logical_and(pred_classes == 1, true_labels == 0).sum()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_value),
        "specificity": float(specificity),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    metrics: Dict[str, float],
    output_path: str,
) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Acc", marker="o")
    plt.plot(val_accs, label="Val Acc", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend(loc="lower right")

    summary_lines = [
        f"Accuracy: {metrics['accuracy'] * 100:.2f}%",
        f"Precision: {metrics['precision'] * 100:.2f}%",
        f"Sensitivity: {metrics['recall'] * 100:.2f}%",
        f"Specificity: {metrics['specificity'] * 100:.2f}%",
        f"F1: {metrics['f1'] * 100:.2f}%",
        f"AUC: {metrics['auc']:.4f}",
    ]
    plt.gcf().text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, class_names: List[str], output_path: str) -> None:
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main experiment routine
# -----------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, labels, file_info, feature_names = load_windowed_features(args.features_dir)
    subject_ids = np.array([info["subject_id"] for info in file_info])

    samples, time_steps, num_channels, num_features = data.shape
    print(f"Loaded data: {samples} samples, {time_steps} windows, {num_channels} channels, {num_features} features")

    # Temporarily disable per-subject standardization to inspect raw feature behavior
    data_raw = data.astype(np.float32, copy=False)

    group_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    train_idx, test_idx = next(group_split.split(data_raw, labels, groups=subject_ids))

    X_train_full, X_test = data_raw[train_idx], data_raw[test_idx]
    y_train_full, y_test = labels[train_idx], labels[test_idx]
    train_subjects = subject_ids[train_idx]

    print(f"Train+Val samples: {len(X_train_full)}, Test samples: {len(X_test)}")

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.random_state)
    train_indices, val_indices = next(val_splitter.split(X_train_full, y_train_full, groups=train_subjects))

    X_train, X_val = X_train_full[train_indices], X_train_full[val_indices]
    y_train, y_val = y_train_full[train_indices], y_train_full[val_indices]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=args.batch_size)

    model_class = MODEL_REGISTRY[args.model]
    model = model_class(num_channels=num_channels, num_features=num_features, time_steps=time_steps, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    results_dir = os.path.join("Comparison", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} - LR {lr:.6f}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))

        print(f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%")
        print(f"Val   Loss {val_loss:.4f}, Val   Acc {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training finished. Best epoch {best_epoch} with val loss {best_val_loss:.4f}")
    model.load_state_dict(best_state)

    print("Evaluating on test set...")
    test_loss, test_probs, test_preds, test_labels, test_acc = evaluate(model, test_loader, criterion, device)
    metrics = compute_binary_metrics(test_labels, test_probs[:, 1])

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(
        f"Precision: {metrics['precision']:.4f}, Sensitivity: {metrics['recall']:.4f}, "
        f"Specificity: {metrics['specificity']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}"
    )

    plot_path = os.path.join(run_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, metrics, plot_path)
    print(f"Training curves saved to {plot_path}")

    class_names = ["Beforedis", "Afterdis"]
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    history = {
        "model": args.model,
        "timestamp": timestamp,
        "features_dir": args.features_dir,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        },
        "test_predictions": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
        "file_info": file_info,
        "feature_names": feature_names,
        "plot_path": plot_path,
        "confusion_matrix_path": cm_path,
    }

    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"History written to {history_path}")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-series model comparison on EEG features")
    parser.add_argument("--features_dir", type=str, default="./EEG_Features_Results_Windowed/", help="Directory containing *_windowed_features.csv files")
    parser.add_argument("--model", type=str, choices=sorted(MODEL_REGISTRY.keys()), default="tcn", help="Model architecture to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay factor")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction from train set")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for deterministic splits")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()
    run_experiment(args)
