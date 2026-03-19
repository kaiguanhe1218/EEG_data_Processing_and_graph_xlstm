import os
import copy
import json
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    auc_value = None
    if len(np.unique(y_true)) == 2:
        try:
            auc_value = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc_value = None

    return {
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': auc_value,
    }


def plot_adjacency_heatmap(adj: np.ndarray, output_path: str, title: str, vmax_abs: float = 1.0):
    mat = np.asarray(adj, dtype=float)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(mat, cmap='coolwarm', vmin=-vmax_abs, vmax=vmax_abs, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_feature_corr_adjacency(x_cf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute channel-channel Pearson correlation over feature dimension.

    x_cf: (C, F)
    returns: (C, C) in roughly [-1, 1]
    """

    x = np.asarray(x_cf, dtype=np.float32)
    x_centered = x - x.mean(axis=1, keepdims=True)
    std = x_centered.std(axis=1, keepdims=True)
    x_norm = x_centered / (std + eps)
    fin = x_norm.shape[1]
    corr = (x_norm @ x_norm.T) / max(fin, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return corr


class MultiScaleFeatureDataset(Dataset):
    """Dataset that returns aligned multi-scale feature tensors."""

    def __init__(self, features, labels):
        if not isinstance(features, (list, tuple)):
            raise TypeError('features must be a sequence of tensors')
        if len(features) == 0:
            raise ValueError('features list cannot be empty')

        sample_counts = {feat.shape[0] for feat in features}
        if len(sample_counts) != 1:
            raise ValueError('all feature tensors must share the same sample count')

        self.features = [torch.as_tensor(feat, dtype=torch.float32) for feat in features]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.features[0].shape[0]

    def __getitem__(self, idx):
        return tuple(feat[idx] for feat in self.features), self.labels[idx]


class GraphLayer(nn.Module):
    def __init__(self, num_channels, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.graph_weights = nn.Parameter(torch.empty(num_channels, num_channels))
        nn.init.xavier_uniform_(self.graph_weights)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        h = self.linear(x)
        weights = torch.softmax(self.graph_weights, dim=-1)
        h = torch.einsum('ij,btjf->btif', weights, h)
        h = self.norm(h)
        return torch.relu(self.dropout(h))


class GraphAttentionLayer(nn.Module):
    """GAT-style graph layer with dynamic edge weights per (batch, time window).

    Input shape: (B, T, C, Fin)
    Output shape: (B, T, C, Fout)
    """

    def __init__(
        self,
        num_channels: int,
        in_features: int,
        out_features: int,
        dropout: float,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # Additive attention: e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
        self.attn_src = nn.Parameter(torch.empty(out_features))
        self.attn_dst = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Fin)
        h = self.linear(x)  # (B, T, C, Fout)

        # Compute attention logits (B, T, C)
        src_scores = torch.einsum('btcf,f->btc', h, self.attn_src)
        dst_scores = torch.einsum('btcf,f->btc', h, self.attn_dst)

        # Broadcast to (B, T, C, C): e_ij = src_i + dst_j
        e = src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2)
        e = self.leaky_relu(e)

        alpha = torch.softmax(e, dim=-1)
        alpha = self.attn_dropout(alpha)

        out = torch.einsum('btij,btjf->btif', alpha, h)
        out = self.norm(out)
        return torch.relu(self.feat_dropout(out))


class GraphCorrelationLayer(nn.Module):
    """Graph layer using analytic correlation-based edge weights per (batch, time window).

    Builds an adjacency matrix A from the current input x by computing Pearson correlation
    between channel feature vectors (over the feature dimension). A is then row-softmaxed
    and used to aggregate transformed node features.

    Input shape: (B, T, C, Fin)
    Output shape: (B, T, C, Fout)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Fin)
        # Build correlation adjacency over Fin (feature dimension).
        x_centered = x - x.mean(dim=-1, keepdim=True)
        std = x_centered.std(dim=-1, keepdim=True)
        x_norm = x_centered / (std + self.eps)

        fin = x_norm.shape[-1]
        corr = torch.einsum('btif,btjf->btij', x_norm, x_norm) / max(fin, 1)

        weights = torch.softmax(corr, dim=-1)
        weights = self.attn_dropout(weights)

        h = self.linear(x)
        out = torch.einsum('btij,btjf->btif', weights, h)
        out = self.norm(out)
        return torch.relu(self.feat_dropout(out))


class XLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.block_gate = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        forward_raw, backward_raw = torch.chunk(lstm_out, 2, dim=-1)

        block_input = torch.cat([forward_raw, backward_raw], dim=-1)
        gates = torch.sigmoid(self.block_gate(block_input))
        gate_forward, gate_backward = torch.chunk(gates, 2, dim=-1)

        mixed_forward = gate_forward * forward_raw + (1.0 - gate_forward) * backward_raw
        mixed_backward = gate_backward * backward_raw + (1.0 - gate_backward) * forward_raw

        block_output = torch.cat([mixed_forward, mixed_backward], dim=-1)
        return self.dropout(block_output)


class XLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        layers = []
        current = input_size
        for layer_idx in range(num_layers):
            layer_dropout = dropout if layer_idx < num_layers - 1 else 0.0
            layers.append(XLSTMLayer(current, hidden_size, layer_dropout))
            current = hidden_size * 2
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class GraphXLSTMBranch(nn.Module):
    def __init__(
        self,
        num_channels,
        num_features,
        hidden_size,
        lstm_layers,
        graph_hidden=128,
        graph_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.graph_layers = nn.ModuleList()
        in_features = num_features
        for _ in range(graph_layers):
            self.graph_layers.append(
                GraphCorrelationLayer(
                    in_features=in_features,
                    out_features=graph_hidden,
                    dropout=dropout,
                )
            )
            in_features = graph_hidden

        self.temporal_encoder = XLSTMEncoder(
            input_size=graph_hidden,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = x
        for layer in self.graph_layers:
            h = layer(h)

        h = h.mean(dim=2)
        temporal = self.temporal_encoder(h)
        attn_scores = self.attention(temporal)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * temporal, dim=1)
        return context, attn_weights


class MultiScaleGraphXLSTM(nn.Module):
    def __init__(self, branch_configs, hidden_size, lstm_layers, dropout=0.3, num_classes=2):
        super().__init__()
        if not branch_configs:
            raise ValueError('branch_configs cannot be empty')

        self.branches = nn.ModuleList([
            GraphXLSTMBranch(
                num_channels=cfg['num_channels'],
                num_features=cfg['num_features'],
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                graph_hidden=cfg.get('graph_hidden', 128),
                graph_layers=cfg.get('graph_layers', 2),
                dropout=dropout,
            )
            for cfg in branch_configs
        ])

        combined_dim = hidden_size * 2 * len(self.branches)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('inputs must be a list of tensors matching the branches')
        if len(inputs) != len(self.branches):
            raise ValueError('input branch count does not match the model configuration')

        contexts = []
        for branch, branch_input in zip(self.branches, inputs):
            context, _ = branch(branch_input)
            contexts.append(context)

        combined = torch.cat(contexts, dim=1)
        return self.classifier(combined)


def load_windowed_features(features_dir, drop_zero_columns=True):
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_windowed_features.csv')]
    if not feature_files:
        raise FileNotFoundError(f'No windowed feature files found in {features_dir}')

    data_list, labels, file_info = [], [], []
    feature_names = None

    id_cols = [
        'filename',
        'subject_id',
        'condition',
        'trial_id',
        'channel',
        'window_index',
        'start_time',
        'end_time',
    ]

    for filename in feature_files:
        file_path = os.path.join(features_dir, filename)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f'Warning: {filename} is empty, skip')
                continue

            current_features = [col for col in df.columns if col not in id_cols]
            if feature_names is None:
                feature_names = current_features
            elif feature_names != current_features:
                raise ValueError(f'Feature columns in {filename} differ from previous files')

            channels = df['channel'].unique()
            windows = sorted(df['window_index'].unique())
            sample_tensor = np.zeros((len(windows), len(channels), len(current_features)), dtype=np.float32)

            for w_idx, window in enumerate(windows):
                for c_idx, channel in enumerate(channels):
                    mask = (df['window_index'] == window) & (df['channel'] == channel)
                    if mask.any():
                        sample_tensor[w_idx, c_idx, :] = df.loc[mask, current_features].values[0]

            data_list.append(sample_tensor)
            label = 1 if 'Afterdis' in filename else 0
            labels.append(label)
            file_info.append({
                'filename': filename,
                'subject_id': str(df['subject_id'].iloc[0]),
                'condition': str(df['condition'].iloc[0]),
                'trial_id': str(df['trial_id'].iloc[0]),
                'label': int(label),
            })
        except Exception as exc:
            print(f'Warning: failed to process {filename}: {exc}, skip')
            continue

    if not data_list:
        raise ValueError(f'Failed to load any feature data from {features_dir}')

    data_array = np.stack(data_list, axis=0)
    labels_array = np.asarray(labels)

    if drop_zero_columns:
        feature_axes = tuple(range(data_array.ndim - 1))
        keep_mask = np.any(np.abs(data_array) > 0, axis=feature_axes)
        if not keep_mask.any():
            raise ValueError(f'All features in {features_dir} are zero, cannot proceed')
        if not keep_mask.all():
            removed = [feature_names[idx] for idx, keep in enumerate(keep_mask) if not keep]
            print(f'{features_dir} removed zero-only features: {removed}')
            data_array = data_array[..., keep_mask]
            feature_names = [feature_names[idx] for idx, keep in enumerate(keep_mask) if keep]

    print(f'{features_dir} shape: {data_array.shape}')
    return data_array, labels_array, file_info, feature_names


def load_multi_scale_feature_sets(feature_dirs):
    datasets = []
    for directory in feature_dirs:
        data, labels, file_info, feature_names = load_windowed_features(directory, drop_zero_columns=True)
        filenames = [info['filename'] for info in file_info]
        datasets.append({
            'directory': directory,
            'data': data,
            'labels': labels,
            'file_info': file_info,
            'feature_names': feature_names,
            'filenames': filenames,
        })

    common_filenames = set(datasets[0]['filenames'])
    for ds in datasets[1:]:
        common_filenames &= set(ds['filenames'])

    if not common_filenames:
        raise ValueError('No common samples across feature directories')

    common_filenames = sorted(common_filenames)
    aligned_data = []
    aligned_labels = None

    for ds in datasets:
        index_map = {fname: idx for idx, fname in enumerate(ds['filenames'])}
        indices = [index_map[fname] for fname in common_filenames]
        aligned_data.append(ds['data'][indices])
        labels_subset = ds['labels'][indices]
        if aligned_labels is None:
            aligned_labels = labels_subset
        else:
            if not np.array_equal(aligned_labels, labels_subset):
                raise ValueError(f'Label mismatch detected in directory {ds["directory"]}')

    base_dataset = datasets[0]
    base_map = {fname: idx for idx, fname in enumerate(base_dataset['filenames'])}
    aligned_info = [base_dataset['file_info'][base_map[fname]] for fname in common_filenames]

    branch_configs = []
    for ds in datasets:
        config = {
            'label': os.path.basename(os.path.normpath(ds['directory'])) or ds['directory'],
            'num_windows': int(ds['data'].shape[1]),
            'num_channels': int(ds['data'].shape[2]),
            'num_features': int(ds['data'].shape[3]),
            'feature_names': ds['feature_names'],
        }
        branch_configs.append(config)

    return aligned_data, aligned_labels, aligned_info, branch_configs


def apply_graph_hparams_to_branch_configs(branch_configs, graph_hidden, graph_layers):
    updated = []
    for cfg in branch_configs:
        cfg = dict(cfg)
        if graph_hidden is not None:
            cfg['graph_hidden'] = int(graph_hidden)
        if graph_layers is not None:
            cfg['graph_layers'] = int(graph_layers)
        updated.append(cfg)
    return updated


def subject_group_standardize(data, subject_ids, eps=1e-6):
    standardized = np.zeros_like(data, dtype=np.float32)
    unique_subjects = np.unique(subject_ids)

    for sid in unique_subjects:
        mask = subject_ids == sid
        subject_tensor = data[mask].astype(np.float32)
        flattened = subject_tensor.reshape(-1, subject_tensor.shape[-1])
        mean = flattened.mean(axis=0, keepdims=True)
        std = flattened.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        normalized = (flattened - mean) / std
        standardized[mask] = normalized.reshape(subject_tensor.shape)

    return standardized


def standardize_branches(branches, subject_ids):
    return [subject_group_standardize(branch, subject_ids) for branch in branches]


def slice_branches(branches, indices):
    return [branch[indices] for branch in branches]


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for features, labels in data_loader:
        inputs = [feat.to(device) for feat in features]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions, predicted_classes, true_labels = [], [], []

    with torch.no_grad():
        for features, labels in data_loader:
            inputs = [feat.to(device) for feat in features]
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            predictions.extend(probs[:, 1].cpu().numpy())
            predicted_classes.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(data_loader), 1)
    predictions = np.array(predictions)
    predicted_classes = np.array(predicted_classes)
    true_labels = np.array(true_labels)
    accuracy = 100.0 * (predicted_classes == true_labels).mean()
    return avg_loss, predictions, predicted_classes, true_labels, accuracy


def plot_results(train_losses, val_losses, train_accs, val_accs, predictions, true_labels, output_path):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()

    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_classes, average='binary', zero_division=0
    )
    auc_value = roc_auc_score(true_labels, predictions)

    metrics_lines = [
        f'Accuracy: {accuracy * 100:.2f}%',
        f'Precision: {precision * 100:.2f}%',
        f'Recall: {recall * 100:.2f}%',
        f'F1-score: {f1 * 100:.2f}%',
        f'AUC: {auc_value:.4f}',
    ]

    plt.gcf().text(
        0.02,
        0.02,
        '\n'.join(metrics_lines),
        ha='left',
        va='bottom',
        fontsize=10,
        bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_feature_importance(model, data_loader, device, branch_configs):
    was_training = model.training
    model.train()

    toggled_modules = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) and module.training:
            module.train(False)
            toggled_modules.append(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.training:
            module.eval()
            toggled_modules.append(module)

    branch_count = len(branch_configs)
    importance = [None] * branch_count
    total_samples = 0

    with torch.enable_grad():
        for features, _ in tqdm(data_loader, desc='Computing feature importance', leave=False):
            batch_size = features[0].shape[0]
            inputs = []
            for feat in features:
                feat = feat.to(device)
                feat.requires_grad_(True)
                inputs.append(feat)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            top_class = torch.argmax(probs, dim=1)
            selected_logits = outputs.gather(1, top_class.unsqueeze(1)).squeeze(1)

            model.zero_grad(set_to_none=True)
            selected_logits.sum().backward()

            for branch_idx, inp in enumerate(inputs):
                if inp.grad is None:
                    continue
                grads = inp.grad.detach().abs().sum(dim=(0, 1, 2))
                if importance[branch_idx] is None:
                    importance[branch_idx] = grads
                else:
                    importance[branch_idx] += grads
                inp.grad = None

            total_samples += batch_size

    for module in toggled_modules:
        module.train(True)

    if not was_training:
        model.eval()

    averaged = []
    for branch_idx, scores in enumerate(importance):
        if scores is None:
            raise RuntimeError(f'Unable to compute feature importance for branch {branch_idx}')
        averaged.append((scores / max(total_samples, 1)).cpu().numpy())
    return averaged


def save_feature_importance(feature_names, feature_scores, output_path):
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_scores,
    })
    df.sort_values('importance', ascending=False, inplace=True)
    df.to_csv(output_path, index=False)


def plot_top_features(feature_names, feature_scores, output_path, top_k=10):
    top_k = min(top_k, len(feature_names))
    indices = np.argsort(feature_scores)[::-1][:top_k]
    top_features = [feature_names[idx] for idx in indices][::-1]
    top_scores = feature_scores[indices][::-1]

    plt.figure(figsize=(8, 6))
    positions = np.arange(len(top_features))
    plt.barh(positions, top_scores, color='teal')
    plt.yticks(positions, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance Score')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    feature_dirs = [
        './EEG_Features_Results_Windowed/',
        './EEG_Features_Results_windowed_0.1/',
    ]
    results_dir = 'EEG_CNN_LSTM/results_graph_xlstm_multiscale'
    models_dir = 'EEG_CNN_LSTM/models_graph_xlstm_multiscale'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, timestamp)
    run_models_dir = os.path.join(models_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_models_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Core hyperparameters ---
    hidden_size = 128
    lstm_layers = 2
    graph_hidden = 96
    graph_layers = 2
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-5
    weight_decay = 5e-3
    dropout = 0.2
    random_state = 42
    patience = 30

    # --- Generalization evaluation (outer CV) ---
    outer_folds = 5
    inner_folds = 4

    print('\nLoading multi-scale feature data...')
    branches, labels, file_info, branch_configs = load_multi_scale_feature_sets(feature_dirs)

    # Ensure graph hyperparameters actually take effect.
    branch_configs = apply_graph_hparams_to_branch_configs(branch_configs, graph_hidden, graph_layers)

    subject_ids = np.array([info['subject_id'] for info in file_info])

    print('\nStandardizing each scale by subject...')
    branches = standardize_branches(branches, subject_ids)

    for cfg, data in zip(branch_configs, branches):
        print(
            f"Scale {cfg['label']}: data shape {data.shape} (windows {cfg['num_windows']} channels {cfg['num_channels']} features {cfg['num_features']})"
        )

    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < outer_folds:
        raise ValueError(f'Need at least {outer_folds} subjects to run {outer_folds}-fold outer CV')

    # --- Save one paper-friendly adjacency example per scale (using sample 0 / window 0)
    # Here adjacency is the feature-correlation matrix used inside GraphCorrelationLayer.
    try:
        example_idx = 0
        example_window = 0
        for scale_idx, cfg in enumerate(branch_configs):
            label = cfg.get('label', f'scale{scale_idx}')
            x_cf = branches[scale_idx][example_idx, example_window]  # (C, F)
            adj_mat = compute_feature_corr_adjacency(x_cf)
            out_path = os.path.join(run_dir, f'adj_example_{label}_sample{example_idx}_win{example_window}.png')
            plot_adjacency_heatmap(
                adj_mat,
                out_path,
                title=f'Adjacency (feature-corr) - {label} - sample {example_idx}, window {example_window}',
                vmax_abs=1.0,
            )
        print(f'Adjacency example plots saved under: {run_dir}')
    except Exception as exc:
        print(f'Warning: failed to plot adjacency examples: {exc}')

    outer_splitter = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    fold_histories = []

    print(f'\nStarting grouped outer CV: {outer_folds} folds')
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        outer_splitter.split(branches[0], labels, groups=subject_ids),
        start=1,
    ):
        print(f'\n===== Outer fold {fold_idx}/{outer_folds} =====')
        set_seed(random_state + fold_idx)

        fold_dir = os.path.join(run_dir, f'fold_{fold_idx:02d}')
        fold_models_dir = os.path.join(run_models_dir, f'fold_{fold_idx:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(fold_models_dir, exist_ok=True)

        X_train_val_branches = slice_branches(branches, train_val_idx)
        y_train_val = labels[train_val_idx]
        train_val_subject_ids = subject_ids[train_val_idx]

        inner_unique_subjects = np.unique(train_val_subject_ids)
        if len(inner_unique_subjects) < inner_folds:
            raise ValueError(
                f'Need at least {inner_folds} subjects inside train/val to run inner split; got {len(inner_unique_subjects)}'
            )

        inner_splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        inner_train_idx, inner_val_idx = next(
            inner_splitter.split(X_train_val_branches[0], y_train_val, groups=train_val_subject_ids)
        )

        train_branches = slice_branches(X_train_val_branches, inner_train_idx)
        val_branches = slice_branches(X_train_val_branches, inner_val_idx)
        test_branches = slice_branches(branches, test_idx)

        y_train = y_train_val[inner_train_idx]
        y_val = y_train_val[inner_val_idx]
        y_test = labels[test_idx]

        print(
            f'Set sizes -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)} '
            f'(subjects train {len(np.unique(train_val_subject_ids[inner_train_idx]))}, '
            f'val {len(np.unique(train_val_subject_ids[inner_val_idx]))}, '
            f'test {len(np.unique(subject_ids[test_idx]))})'
        )

        train_dataset = MultiScaleFeatureDataset(train_branches, y_train)
        val_dataset = MultiScaleFeatureDataset(val_branches, y_val)
        test_dataset = MultiScaleFeatureDataset(test_branches, y_test)

        train_batch_size = min(batch_size, len(train_dataset))
        val_batch_size = max(1, min(batch_size, len(val_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = MultiScaleGraphXLSTM(
            branch_configs=branch_configs,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            num_classes=2,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        patience_counter = 0

        best_model_path = os.path.join(fold_models_dir, f'best_graph_xlstm_model_fold{fold_idx:02d}_{timestamp}.pth')
        history_path = os.path.join(fold_dir, f'graph_xlstm_history_fold{fold_idx:02d}_{timestamp}.json')
        plot_path = os.path.join(fold_dir, f'graph_xlstm_training_fold{fold_idx:02d}_{timestamp}.png')

        print('Starting training...')
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _, val_acc = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] lr={current_lr:.6f} '
                    f'train_loss={train_loss:.4f} train_acc={train_acc:.2f}% '
                    f'val_loss={val_loss:.4f} val_acc={val_acc:.2f}%'
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                torch.save(best_model_state, best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(val_loss)

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        model.load_state_dict(best_model_state)

        test_loss, test_predictions, test_classes, test_labels, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )
        metrics = compute_binary_metrics(test_labels, test_predictions)
        metrics.update({
            'fold': int(fold_idx),
            'test_loss': float(test_loss),
            'test_accuracy_pct': float(test_accuracy),
        })

        print('Fold test metrics:', metrics)
        print(classification_report(test_labels, test_classes, target_names=['Beforedis', 'Afterdis']))

        plot_results(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            test_predictions,
            test_labels,
            plot_path,
        )

        history = {
            'fold': int(fold_idx),
            'best_epoch': int(best_epoch),
            'best_val_loss': float(best_val_loss),
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'train_accuracies': [float(x) for x in train_accuracies],
            'val_accuracies': [float(x) for x in val_accuracies],
            'test_predictions': test_predictions.tolist(),
            'test_labels': [int(lbl) for lbl in test_labels],
            'metrics': metrics,
            'branch_configs': branch_configs,
            'best_model_path': best_model_path,
            'plot_path': plot_path,
        }

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        fold_metrics.append(metrics)
        fold_histories.append(history_path)

    def _mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    acc_mean, acc_std = _mean_std([m.get('accuracy') for m in fold_metrics])
    f1_mean, f1_std = _mean_std([m.get('f1') for m in fold_metrics])
    auc_mean, auc_std = _mean_std([m.get('auc') for m in fold_metrics])

    summary = {
        'outer_folds': int(outer_folds),
        'inner_folds': int(inner_folds),
        'random_state': int(random_state),
        'hyperparams': {
            'hidden_size': int(hidden_size),
            'lstm_layers': int(lstm_layers),
            'graph_hidden': int(graph_hidden),
            'graph_layers': int(graph_layers),
            'batch_size': int(batch_size),
            'num_epochs': int(num_epochs),
            'learning_rate': float(learning_rate),
            'weight_decay': float(weight_decay),
            'dropout': float(dropout),
            'patience': int(patience),
        },
        'metrics_per_fold': fold_metrics,
        'aggregate': {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
        },
        'fold_history_paths': fold_histories,
    }

    summary_path = os.path.join(run_dir, f'cv_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n===== CV Summary =====')
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}" if acc_mean is not None else 'Accuracy: n/a')
    print(f"F1:       {f1_mean:.4f} ± {f1_std:.4f}" if f1_mean is not None else 'F1: n/a')
    print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}" if auc_mean is not None else 'AUC: n/a')
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
