# -*- coding: utf-8 -*-
"""
Multimodal VAE for Ca2+ event embeddings.

Public-release conventions:
- English-only comments and messages.
- No personal / lab-specific paths. Use relative, anonymized paths.
- Commented-out code removed to keep the script clean.

Model summary (aligned with the manuscript description):
- Inputs per event:
  1) ΔF/F waveform: standardized per event (zero mean, unit variance), zero-padded to a fixed length,
     plus a binary mask indicating valid time points.
  2) 19 structured features: z-scored before encoding.
- Architecture:
  - Waveform encoder: temporal conv + biLSTM + attention-style weighting.
  - Structured encoder: MLP.
  - Attention-based fusion combines both modalities.
  - Latent distribution: mean and log-variance vectors (dim = 64).
  - Decoder reconstructs waveform (tanh) and structured features (linear).
- Training:
  - Adam, lr=1e-3, batch=128, up to 500 epochs, early stopping on validation loss.
  - Loss = masked MSE (waveform) + MSE (features) + KL divergence.
- Inference:
  - Use z_mean (latent mean) as the final 64-d embedding for each event.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# I/O (anonymized + portable)
# -----------------------------
DATA_DIR = Path("./data")
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOMA_PATHS_TXT = DATA_DIR / "soma_paths.txt"
MAT_PATHS_TXT = DATA_DIR / "aqua_paths.txt"
EVENT_TABLE_CSV = DATA_DIR / "events_with_location_labels.csv"

WAVE_PKL = OUT_DIR / "all_event_waveforms.pkl"
BEST_MODEL_PTH = OUT_DIR / "best_multimodal_vae.pth"
Z_PKL = OUT_DIR / "vae_event_embeddings.pkl"
PAIRWISE_PKL = OUT_DIR / "pairwise_similarity_by_cell.pkl"


# -----------------------------
# Waveform extraction utilities
# -----------------------------
def load_paths_from_txt(txt_path: Path) -> List[str]:
    """
    Load one path per line. Each line should be a Python-style quoted string.
    Uses ast.literal_eval() for safety.
    """
    paths: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(ast.literal_eval(line))
    return paths


def _process_event_coords(i: int, coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """
    Extract start/end frame from event voxel coordinates returned by np.unravel_index().
    coords: (x, y, t), each array shape (n_voxels,).
    """
    x, y, t = coords
    min_frame = int(t.min())
    return {
        "start_point_x": float(np.median(x[t == min_frame])),
        "start_point_y": float(np.median(y[t == min_frame])),
        "event_start_frame": min_frame,
        "event_end_frame": int(t.max()),
        "order": i,
    }


def extract_event_waveforms(
    aqua_mat_path: str,
    event_indices: np.ndarray,
    min_pad: int = 3,
    max_frame: int = 2999,
) -> List[np.ndarray]:
    """
    Extract ΔF/F waveforms from AQuA outputs for a subset of events.

    Notes:
    - Uses event start/end frames from voxel coordinates.
    - Expands very short events by +/- min_pad frames (clipped to [0, max_frame]).
    - Returns a list of 1D numpy arrays (variable length).
    """
    x3d = sio.loadmat(aqua_mat_path.replace("_AQuA.mat", "_res_fts_loc_x3D.mat"))["x3D"]
    sz = sio.loadmat(aqua_mat_path.replace("_AQuA.mat", "_res_opts_sz.mat"))["sz"][0]
    dff = sio.loadmat(aqua_mat_path.replace("_AQuA.mat", "_res_dffMatFilter.mat"))["dffMatFilter"]

    coords_all = [
        np.unravel_index(x3d[0, i].astype(int), sz.astype(int), order="F")
        for i in range(x3d.shape[1])
    ]
    coords_all = [coords_all[i] for i in event_indices]

    events = Parallel(n_jobs=-1)(
        delayed(_process_event_coords)(i, coords) for i, coords in enumerate(coords_all)
    )
    events_df = pd.DataFrame(events)

    waveforms: List[np.ndarray] = []
    for idx, row in events_df.iterrows():
        start = int(row["event_start_frame"])
        end = int(row["event_end_frame"])

        if (end - start) < 3:
            start = max(0, start - min_pad)
            end = min(max_frame, end + min_pad)

        # dff shape is typically (n_events, n_frames, 1) in many AQuA exports
        wave = dff[idx, start:end, 0].astype(np.float32)
        waveforms.append(wave)

    return waveforms


# -----------------------------
# Preprocessing (aligned w/ description)
# -----------------------------
def standardize_and_pad_waveforms(
    waveforms: List[np.ndarray],
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-event standardize (zero mean, unit variance), then zero-pad to target_length.
    Also returns a binary mask of valid time points.

    Returns:
      padded_wave: (N, T) float32
      mask:        (N, T) float32  (1=valid, 0=padding)
    """
    padded = np.zeros((len(waveforms), target_length), dtype=np.float32)
    mask = np.zeros((len(waveforms), target_length), dtype=np.float32)

    for i, w in enumerate(waveforms):
        w = w.astype(np.float32)
        mu = float(w.mean())
        sd = float(w.std()) + 1e-8
        w_std = (w - mu) / sd

        L = min(len(w_std), target_length)
        padded[i, :L] = w_std[:L]
        mask[i, :L] = 1.0

    return padded, mask


def zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise z-score with safe handling for zero-std columns (drop them).
    """
    stds = df.std()
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print(f"[Info] Dropping zero-std feature columns: {zero_std_cols}")
        df = df.drop(columns=zero_std_cols)

    return (df - df.mean()) / (df.std() + 1e-8)


# -----------------------------
# VAE modules (PyTorch)
# -----------------------------
class WaveEncoder(nn.Module):
    """
    Waveform encoder: conv -> biLSTM -> attention-like weighting -> pooled features -> projection.
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_ln = nn.LayerNorm(128)
        self.lstm_dropout = nn.Dropout(0.4)

        self.channel_attn = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, latent_dim),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: (B, T)
        x = wave.unsqueeze(1)  # (B, 1, T)
        x = self.conv(x)       # (B, 128, T//4)
        x = x.permute(0, 2, 1) # (B, T//4, 128)

        x, _ = self.lstm(x)    # (B, T//4, 128)
        x = self.lstm_ln(x)
        x = self.lstm_dropout(x)

        w = self.channel_attn(x.mean(dim=1))     # (B, 128)
        x = x * w.unsqueeze(1)                   # (B, T//4, 128)

        x = x.permute(0, 2, 1)                   # (B, 128, T//4)
        x = self.post(x).squeeze(-1)             # (B, 128)

        return self.proj(x)                      # (B, latent_dim)


class StructEncoder(nn.Module):
    """
    Structured feature encoder: MLP.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion that adaptively weights waveform vs structured signals per sample.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.sample_attn = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.feat_transform = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, wave_feat: torch.Tensor, struct_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([wave_feat, struct_feat], dim=-1)
        sample_w = self.sample_attn(combined)          # (B, 1)
        transformed = self.feat_transform(combined)    # (B, latent_dim)

        fused = sample_w * wave_feat + (1 - sample_w) * transformed
        return self.alpha * fused + (1 - self.alpha) * struct_feat


class MultiModalVAE(nn.Module):
    """
    Multimodal VAE:
      - Encoders per modality
      - Fusion -> z_mean, z_log_var
      - Reparameterization (with dropout regularization)
      - Decoder reconstructs waveform (tanh) + structured features (linear)
    """

    def __init__(self, wave_dim: int, struct_dim: int, latent_dim: int):
        super().__init__()
        self.wave_dim = wave_dim
        self.struct_dim = struct_dim
        self.latent_dim = latent_dim

        self.wave_encoder = WaveEncoder(latent_dim)
        self.struct_encoder = StructEncoder(struct_dim, latent_dim)
        self.fusion = AttentionFusion(latent_dim)

        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)
        self.z_dropout = nn.Dropout(0.2)

        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_ln1 = nn.LayerNorm(512)
        self.dec_fc2 = nn.Linear(512, 1024)
        self.dec_ln2 = nn.LayerNorm(1024)
        self.dec_fc3 = nn.Linear(1024, wave_dim + struct_dim)
        self.relu = nn.ReLU()
        self.dec_dropout = nn.Dropout(0.3)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return self.z_dropout(z)

    def encode(self, wave: torch.Tensor, struct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wave_feat = self.wave_encoder(wave)
        struct_feat = self.struct_encoder(struct)
        fused = self.fusion(wave_feat, struct_feat)
        return self.z_mean(fused), self.z_log_var(fused)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dec_fc1(z)
        x = self.dec_ln1(x)
        x = self.relu(x)
        x = self.dec_dropout(x)

        x = self.dec_fc2(x)
        x = self.dec_ln2(x)
        x = self.relu(x)
        x = self.dec_dropout(x)

        x = self.dec_fc3(x)

        wave_recon = torch.tanh(x[:, : self.wave_dim])
        struct_recon = x[:, self.wave_dim :]
        return wave_recon, struct_recon

    def forward(self, wave: torch.Tensor, struct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(wave, struct)
        z = self.reparameterize(mean, log_var)
        wave_recon, struct_recon = self.decode(z)
        return wave_recon, struct_recon, mean, log_var


def masked_mse(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean squared error for waveform reconstruction.
    """
    se = (y_true - y_pred) ** 2
    valid = mask.sum(dim=1).clamp(min=1.0)
    return (se * mask).sum(dim=1).div(valid).mean()


def kl_divergence(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Standard VAE KL divergence term.
    """
    return -0.5 * torch.mean(1 + log_var - mean.pow(2) - torch.exp(log_var))


@torch.no_grad()
def get_latent_means(model: MultiModalVAE, wave: torch.Tensor, struct: torch.Tensor) -> np.ndarray:
    """
    Inference: return z_mean as the final embedding (shape: N x latent_dim).
    """
    model.eval()
    z_mean, _ = model.encode(wave, struct)
    return z_mean.detach().cpu().numpy()


# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running on device: {device}")

    # 1) Load metadata for mapping events to cells
    meta = pd.read_csv(EVENT_TABLE_CSV, usecols=["Starting_Frame", "mouse_id", "cell_id", "wt"])
    cell_ids = meta["cell_id"].drop_duplicates().reset_index(drop=True)

    # 2) Load AQuA paths (kept as a list; content should be anonymized in the txt files)
    soma_paths = load_paths_from_txt(SOMA_PATHS_TXT)
    aqua_paths = load_paths_from_txt(MAT_PATHS_TXT)

    # 3) Extract all waveforms in the same order as the event table per cell
    all_waveforms: List[np.ndarray] = []
    for i in tqdm(range(len(aqua_paths)), desc="Extracting waveforms"):
        cell_id = cell_ids.iloc[i]
        cell_rows = meta[meta["cell_id"] == cell_id].reset_index(drop=True)
        event_indices = cell_rows.index.values  # assumes AQuA event ordering matches CSV ordering per cell

        waves = extract_event_waveforms(aqua_paths[i], event_indices=event_indices)
        all_waveforms.extend(waves)

    with open(WAVE_PKL, "wb") as f:
        pickle.dump({"waveforms": all_waveforms}, f)
    print(f"[Info] Saved waveforms to: {WAVE_PKL}")

    # 4) Load structured features (19 columns) and z-score
    feature_cols = [
        "Basic_Area",
        "Basic_Perimeter",
        "Basic_Circularity",
        "Curve_Max_Dff",
        "Curve_Duration_50_to_50",
        "Curve_Duration_10_to_10",
        "Curve_Rising_duration_10_to_90",
        "Curve_Decaying_duration_90_to_10",
        "Curve_dat_AUC",
        "Curve_dff_AUC",
        "Propagation_onset_overall",
        "Propagation_offset_overall",
        "Landmark_event_toward_landmark_landmark_1",
        "Landmark_event_away_from_landmark_landmark_1",
        "Landmark_event_toward_landmark_before_reaching_landmark_1",
        "Landmark_event_away_from_landmark_after_reaching_landmark_1",
        "Network_Temporal_density",
        "Network_Temporal_density_with_similar_size_only",
        "Network_Spatial_density",
    ]
    struct_df = pd.read_csv(EVENT_TABLE_CSV, usecols=feature_cols)
    struct_df = zscore_features(struct_df)

    # 5) Load waveforms and pad to a fixed length
    with open(WAVE_PKL, "rb") as f:
        waveforms = pickle.load(f)["waveforms"]

    lengths = np.array([len(w) for w in waveforms], dtype=int)
    # Fixed length choice: 95th percentile to reduce extreme-padding, then clip longer traces.
    target_len = int(np.quantile(lengths, 0.95))
    print(f"[Info] Waveform target length: {target_len}")

    padded_wave, wave_mask = standardize_and_pad_waveforms(waveforms, target_length=target_len)

    # 6) Train/val split
    X_train, X_val, S_train, S_val, M_train, M_val = train_test_split(
        padded_wave,
        struct_df.values.astype(np.float32),
        wave_mask,
        test_size=0.2,
        random_state=42,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    S_train_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    S_val_t = torch.tensor(S_val, dtype=torch.float32, device=device)
    M_train_t = torch.tensor(M_train, dtype=torch.float32, device=device)
    M_val_t = torch.tensor(M_val, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train_t, S_train_t, M_train_t)
    val_ds = TensorDataset(X_val_t, S_val_t, M_val_t)

    # 7) Build VAE (latent dim = 64)
    latent_dim = 64
    model = MultiModalVAE(
        wave_dim=X_train_t.shape[1],
        struct_dim=S_train_t.shape[1],
        latent_dim=latent_dim,
    ).to(device)

    # 8) Train (aligned with description)
    lr = 1e-3
    batch_size = 128
    max_epochs = 500
    patience = 20  # early stopping patience (validation loss)
    min_delta = 1e-4

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    no_improve = 0

    # Loss weights (kept simple and consistent with the description)
    wave_weight = 1.0
    struct_weight = 1.0
    beta = 1.0

    print("[Info] Training VAE...")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for xb, sb, mb in train_loader:
            optimizer.zero_grad()

            wave_recon, struct_recon, z_mean, z_log_var = model(xb, sb)

            loss_wave = masked_mse(xb, wave_recon, mb)
            loss_struct = F.mse_loss(sb, struct_recon)
            loss_kl = kl_divergence(z_mean, z_log_var)

            loss = wave_weight * loss_wave + struct_weight * loss_struct + beta * loss_kl
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, sb, mb in val_loader:
                wave_recon, struct_recon, z_mean, z_log_var = model(xb, sb)

                loss_wave = masked_mse(xb, wave_recon, mb)
                loss_struct = F.mse_loss(sb, struct_recon)
                loss_kl = kl_divergence(z_mean, z_log_var)

                loss = wave_weight * loss_wave + struct_weight * loss_struct + beta * loss_kl
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"[Epoch {epoch+1:03d}] train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PTH)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Info] Early stopping at epoch {epoch+1}. Best val={best_val:.4f}")
                break

    print(f"[Info] Best model checkpoint: {BEST_MODEL_PTH}")

    # 9) Inference: encode all events using z_mean
    model.load_state_dict(torch.load(BEST_MODEL_PTH, map_location=device))
    X_all_t = torch.tensor(padded_wave, dtype=torch.float32, device=device)
    S_all_t = torch.tensor(struct_df.values.astype(np.float32), dtype=torch.float32, device=device)

    z_mean_all = get_latent_means(model, X_all_t, S_all_t)
    print(f"[Info] Latent embeddings shape: {z_mean_all.shape}")

    with open(Z_PKL, "wb") as f:
        pickle.dump({"z_mean": z_mean_all}, f)
    print(f"[Info] Saved embeddings to: {Z_PKL}")

    # 10) Pairwise similarity within each cell (cosine / pearson / euclidean / manhattan / spearman)
    from scipy.stats import rankdata
    from sklearn.preprocessing import StandardScaler
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar

    idx_df = pd.read_csv(EVENT_TABLE_CSV, usecols=["cell_id", "Starting_Frame"]).reset_index(drop=True)
    scaler = StandardScaler()
    z_norm = scaler.fit_transform(z_mean_all)
    idx_df["z"] = list(z_norm)

    def process_cell_group(cell_id: int, group_df: pd.DataFrame) -> pd.DataFrame:
        V = np.vstack(group_df["z"].values)  # (n, d)
        n = V.shape[0]
        if n < 2:
            return pd.DataFrame()

        norms = np.linalg.norm(V, axis=1)
        dot = V @ V.T
        cos = dot / (norms[:, None] * norms[None, :] + 1e-8)

        means = V.mean(axis=1, keepdims=True)
        X = V - means
        stds = np.linalg.norm(X, axis=1)
        denom = stds[:, None] * stds[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson = np.where(denom == 0, np.nan, (X @ X.T) / denom)

        sq = np.sum(V**2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2 * dot
        dist = np.sqrt(np.maximum(dist_sq, 0))
        euclid = 1.0 / (1.0 + dist)

        manhattan_dist = np.sum(np.abs(V[:, None, :] - V[None, :, :]), axis=2)
        manhattan = 1.0 / (1.0 + manhattan_dist)

        ranks = np.apply_along_axis(rankdata, 1, V)
        R = ranks - ranks.mean(axis=1, keepdims=True)
        Rstd = np.linalg.norm(R, axis=1)
        denom_s = Rstd[:, None] * Rstd[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            spearman = np.where(denom_s == 0, np.nan, (R @ R.T) / denom_s)

        triu = np.triu_indices(n, k=1)
        return pd.DataFrame(
            {
                "index_a": triu[0],
                "index_b": triu[1],
                "cell_id": cell_id,
                "cosine_corr": cos[triu],
                "pearson_corr": pearson[triu],
                "euclidean_sim": euclid[triu],
                "manhattan_sim": manhattan[triu],
                "spearman_corr": spearman[triu],
            }
        )

    groups = [(cid, g) for cid, g in idx_df.groupby("cell_id")]
    tasks = [delayed(process_cell_group)(cid, g) for cid, g in groups]

    with ProgressBar():
        out = compute(*tasks)

    pairwise_df = pd.concat(out, ignore_index=True)
    with open(PAIRWISE_PKL, "wb") as f:
        pickle.dump({"pairwise": pairwise_df}, f)

    print(f"[Info] Saved pairwise similarity to: {PAIRWISE_PKL}")
    print(pairwise_df.head())
