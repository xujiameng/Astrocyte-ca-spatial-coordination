# -*- coding: utf-8 -*-
 

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Optional DTW (used only for evaluation)
try:
    from dtaidistance import dtw
    _HAS_DTW = True
except Exception:
    _HAS_DTW = False


# -----------------------------
# Global config (MATCHES TEXT)
# -----------------------------
SEED = 42
LATENT_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 500

# Early stopping (validation loss)
PATIENCE = 20
MIN_DELTA = 1e-4

# Loss weights (description implies a simple sum; keep = 1.0)
WAVE_WEIGHT = 1.0
STRUCT_WEIGHT = 1.0
BETA_KL = 1.0

# -----------------------------
# Paths (anonymized)
# -----------------------------
DATA_DIR = Path("./data")
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_TABLE_CSV = DATA_DIR / "events_with_features.csv"
WAVE_PKL = DATA_DIR / "all_event_waveforms.pkl"  # expects {'signal_list': [...]}
CHECKPOINT_PATH = OUT_DIR / "best_multimodal_vae.pth"
MODEL_BUNDLE_PATH = OUT_DIR / "multimodal_vae_model_bundle.pth"

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Preprocessing (MATCHES TEXT)
# -----------------------------
def zscore_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column. Drop any zero-std columns (rare, but safe).
    """
    stds = df.std()
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print(f"[Info] Dropping zero-std columns: {zero_std_cols}")
        df = df.drop(columns=zero_std_cols)
    return (df - df.mean()) / (df.std() + 1e-8)


def standardize_per_event_and_pad(
    wave_list: list[np.ndarray],
    target_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per event: standardize to zero mean, unit variance, then zero-pad to target_length.
    Also return a binary mask for valid time points (1=valid, 0=padding).
    """
    n = len(wave_list)
    padded = np.zeros((n, target_length), dtype=np.float32)
    mask = np.zeros((n, target_length), dtype=np.float32)

    for i, w in enumerate(wave_list):
        w = np.asarray(w, dtype=np.float32)
        mu = float(w.mean())
        sd = float(w.std()) + 1e-8
        w_std = (w - mu) / sd

        L = min(len(w_std), target_length)
        padded[i, :L] = w_std[:L]
        mask[i, :L] = 1.0

    return padded, mask


# -----------------------------
# Model components (MATCHES TEXT)
# -----------------------------
class WaveEncoder(nn.Module):
    """
    Waveform feature extractor:
    temporal convolution + recurrent modeling (biLSTM) + attention-style channel weighting.
    """
    def __init__(self, latent_dim: int):
        super().__init__()

        self.conv_block = nn.Sequential(
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
            bidirectional=True,
            batch_first=True
        )
        self.lstm_ln = nn.LayerNorm(128)
        self.lstm_dropout = nn.Dropout(0.4)

        self.channel_attn = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )

        self.post_pool = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = x.unsqueeze(1)          # (B, 1, T)
        x = self.conv_block(x)      # (B, 128, T//4)

        x = x.permute(0, 2, 1)      # (B, T//4, 128)
        x, _ = self.lstm(x)         # (B, T//4, 128)
        x = self.lstm_ln(x)
        x = self.lstm_dropout(x)

        w = self.channel_attn(x.mean(dim=1))    # (B, 128)
        x = x * w.unsqueeze(1)

        x = x.permute(0, 2, 1)      # (B, 128, T//4)
        x = self.post_pool(x).squeeze(-1)  # (B, 128)

        return self.proj(x)         # (B, latent_dim)


class StructEncoder(nn.Module):
    """
    Structured feature encoder (MLP).
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion that adaptively weights waveform- and feature-derived embeddings.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.sample_attn = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.feat_transform = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, wave_feat: torch.Tensor, struct_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([wave_feat, struct_feat], dim=-1)
        sample_w = self.sample_attn(combined)          # (B, 1)
        transformed = self.feat_transform(combined)    # (B, latent_dim)

        fused = sample_w * wave_feat + (1 - sample_w) * transformed
        return self.alpha * fused + (1 - self.alpha) * struct_feat


class MultiModalVAE(nn.Module):
    """
    Multimodal VAE with:
    - modality-specific encoders
    - attention fusion
    - latent mean/logvar (64-d)
    - reparameterization with dropout regularization
    - decoders: waveform (tanh), structured (linear)
    """
    def __init__(self, wave_dim: int, struct_dim: int, latent_dim: int = 64):
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

        # Decoder heads
        self.wave_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, wave_dim)
        )

        self.struct_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, struct_dim)
        )

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return self.z_dropout(z)

    def encode(self, wave: torch.Tensor, struct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_feat = self.wave_encoder(wave)
        s_feat = self.struct_encoder(struct)
        fused = self.fusion(w_feat, s_feat)
        return self.z_mean(fused), self.z_log_var(fused)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wave_recon = torch.tanh(self.wave_decoder(z))  # tanh for waveform
        struct_recon = self.struct_decoder(z)          # linear for struct
        return wave_recon, struct_recon

    def forward(self, wave: torch.Tensor, struct: torch.Tensor):
        z_mean, z_log_var = self.encode(wave, struct)
        z = self.reparameterize(z_mean, z_log_var)
        wave_recon, struct_recon = self.decode(z)
        return wave_recon, struct_recon, z_mean, z_log_var


# -----------------------------
# Losses (MATCHES TEXT)
# -----------------------------
def masked_mse(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    se = (y_true - y_pred) ** 2
    valid = mask.sum(dim=1).clamp(min=1.0)
    return (se * mask).sum(dim=1).div(valid).mean()


def kl_divergence(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))


# -----------------------------
# Training utils (EARLY STOPPING ON VAL)
# -----------------------------
def train_vae(
    model: MultiModalVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[MultiModalVAE, Dict[str, list], Dict[str, float]]:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_recon_wave": [],
        "val_recon_wave": [],
        "train_recon_struct": [],
        "val_recon_struct": [],
        "train_kl": [],
        "val_kl": [],
    }

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ---- train ----
        model.train()
        tr_total = tr_w = tr_s = tr_kl = 0.0

        for wave_b, struct_b, mask_b in train_loader:
            wave_b = wave_b.to(device)
            struct_b = struct_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad()
            wave_recon, struct_recon, z_mean, z_log_var = model(wave_b, struct_b)

            loss_wave = masked_mse(wave_b, wave_recon, mask_b)
            loss_struct = F.mse_loss(struct_b, struct_recon)
            loss_kl = kl_divergence(z_mean, z_log_var)

            loss = WAVE_WEIGHT * loss_wave + STRUCT_WEIGHT * loss_struct + BETA_KL * loss_kl
            loss.backward()
            optimizer.step()

            bs = wave_b.size(0)
            tr_total += loss.item() * bs
            tr_w += loss_wave.item() * bs
            tr_s += loss_struct.item() * bs
            tr_kl += loss_kl.item() * bs

        n_tr = len(train_loader.dataset)
        tr_total /= n_tr
        tr_w /= n_tr
        tr_s /= n_tr
        tr_kl /= n_tr

        # ---- val ----
        model.eval()
        va_total = va_w = va_s = va_kl = 0.0

        with torch.no_grad():
            for wave_b, struct_b, mask_b in val_loader:
                wave_b = wave_b.to(device)
                struct_b = struct_b.to(device)
                mask_b = mask_b.to(device)

                wave_recon, struct_recon, z_mean, z_log_var = model(wave_b, struct_b)

                loss_wave = masked_mse(wave_b, wave_recon, mask_b)
                loss_struct = F.mse_loss(struct_b, struct_recon)
                loss_kl = kl_divergence(z_mean, z_log_var)

                loss = WAVE_WEIGHT * loss_wave + STRUCT_WEIGHT * loss_struct + BETA_KL * loss_kl

                bs = wave_b.size(0)
                va_total += loss.item() * bs
                va_w += loss_wave.item() * bs
                va_s += loss_struct.item() * bs
                va_kl += loss_kl.item() * bs

        n_va = len(val_loader.dataset)
        va_total /= n_va
        va_w /= n_va
        va_s /= n_va
        va_kl /= n_va

        # log
        history["train_total_loss"].append(tr_total)
        history["val_total_loss"].append(va_total)
        history["train_recon_wave"].append(tr_w)
        history["val_recon_wave"].append(va_w)
        history["train_recon_struct"].append(tr_s)
        history["val_recon_struct"].append(va_s)
        history["train_kl"].append(tr_kl)
        history["val_kl"].append(va_kl)

        print(
            f"[Epoch {epoch+1:03d}/{MAX_EPOCHS}] "
            f"train={tr_total:.4f} (wave={tr_w:.4f}, struct={tr_s:.4f}, kl={tr_kl:.4f}) | "
            f"val={va_total:.4f} (wave={va_w:.4f}, struct={va_s:.4f}, kl={va_kl:.4f})"
        )

        # early stopping on validation loss
        if va_total < best_val - MIN_DELTA:
            best_val = va_total
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CHECKPOINT_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[Info] Early stopping at epoch {epoch+1}. Best val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_val_total": best_val,
        "final_val_wave": history["val_recon_wave"][-1],
        "final_val_struct": history["val_recon_struct"][-1],
        "final_val_kl": history["val_kl"][-1],
    }
    return model, history, metrics


# -----------------------------
# Inference (MATCHES TEXT)
# -----------------------------
@torch.no_grad()
def get_latent_mean_embedding(model: MultiModalVAE, wave: torch.Tensor, struct: torch.Tensor) -> np.ndarray:
    model.eval()
    z_mean, _ = model.encode(wave, struct)
    return z_mean.detach().cpu().numpy()


# -----------------------------
# Evaluation helpers (kept, but cleaned)
# -----------------------------
def calculate_dtw_distance(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:
    if not _HAS_DTW:
        return float("nan")
    valid_idx = np.where(mask == 1)[0]
    if len(valid_idx) < 2:
        return float("nan")
    return float(dtw.distance(original[valid_idx], reconstructed[valid_idx]))


def calculate_cosine_similarity(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:
    valid_idx = np.where(mask == 1)[0]
    if len(valid_idx) < 2:
        return float("nan")
    return float(1.0 - cosine(original[valid_idx], reconstructed[valid_idx]))


@torch.no_grad()
def calculate_reconstruction_metrics(
    model: MultiModalVAE,
    wave_data: torch.Tensor,
    struct_data: torch.Tensor,
    mask_data: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    wave_data = wave_data.to(device)
    struct_data = struct_data.to(device)
    mask_data = mask_data.to(device)

    wave_recon, struct_recon, _, _ = model(wave_data, struct_data)

    wave_mse = masked_mse(wave_data, wave_recon, mask_data).item()

    abs_err = torch.abs(wave_data - wave_recon)
    valid = mask_data.sum(dim=1).clamp(min=1.0)
    wave_mae = (abs_err * mask_data).sum(dim=1).div(valid).mean().item()

    struct_mse = F.mse_loss(struct_data, struct_recon).item()
    struct_mae = F.l1_loss(struct_data, struct_recon).item()

    corr = np.corrcoef(
        struct_data.detach().cpu().numpy().flatten(),
        struct_recon.detach().cpu().numpy().flatten()
    )[0, 1]

    return {
        "wave_mse_masked": wave_mse,
        "wave_mae_masked": wave_mae,
        "struct_mse": struct_mse,
        "struct_mae": struct_mae,
        "struct_flat_corr": float(corr),
    }


def plot_loss_curves(history: Dict[str, list]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_total_loss"], label="Train")
    axes[0, 0].plot(history["val_total_loss"], label="Val")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["train_recon_wave"], label="Train")
    axes[0, 1].plot(history["val_recon_wave"], label="Val")
    axes[0, 1].set_title("Wave reconstruction loss (masked MSE)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history["train_recon_struct"], label="Train")
    axes[1, 0].plot(history["val_recon_struct"], label="Val")
    axes[1, 0].set_title("Structured feature reconstruction loss (MSE)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["train_kl"], label="Train")
    axes[1, 1].plot(history["val_kl"], label="Val")
    axes[1, 1].set_title("KL divergence")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_wave_reconstruction(
    model: MultiModalVAE,
    wave_data: torch.Tensor,
    struct_data: torch.Tensor,
    mask_data: torch.Tensor,
    device: torch.device,
    save_path: str | None = None,
    num_samples: int = 3,
    specific_ids: list[int] | None = None,
    show_legend: bool = True,
) -> None:
    model.eval()

    n = len(wave_data)
    if specific_ids is not None:
        indices = np.array([i for i in specific_ids if 0 <= i < n], dtype=int)
        if len(indices) == 0:
            print("[Warn] No valid ids in specific_ids.")
            return
    else:
        indices = np.random.choice(n, size=min(num_samples, n), replace=False)

    wave_s = wave_data[indices].to(device)
    struct_s = struct_data[indices].to(device)
    mask_s = mask_data[indices].to(device)

    wave_recon, _, _, _ = model(wave_s, struct_s)

    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 3 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        orig = wave_s[i].detach().cpu().numpy()
        rec = wave_recon[i].detach().cpu().numpy()
        mask = mask_s[i].detach().cpu().numpy()

        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) > 0:
            start = valid_idx[0]
            end = valid_idx[-1] + 1
        else:
            start, end = 0, len(orig)

        t = np.arange(start, end)
        axes[i].plot(t, orig[start:end], label="Original", linewidth=2)
        axes[i].plot(t, rec[start:end], label="Reconstructed", linewidth=2, linestyle="--")

        mse = float(np.mean((orig[valid_idx] - rec[valid_idx]) ** 2)) if len(valid_idx) else float(np.mean((orig - rec) ** 2))
        mae = float(np.mean(np.abs(orig[valid_idx] - rec[valid_idx]))) if len(valid_idx) else float(np.mean(np.abs(orig - rec)))
        cos_sim = calculate_cosine_similarity(orig, rec, mask)
        dtw_dist = calculate_dtw_distance(orig, rec, mask)

        title = f"Sample {idx} | MSE={mse:.4f}, MAE={mae:.4f}, Cos={cos_sim:.4f}"
        if _HAS_DTW:
            title += f", DTW={dtw_dist:.4f}"
        axes[i].set_title(title)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("ΔF/F (standardized)")
        axes[i].grid(alpha=0.3)
        if show_legend:
            axes[i].legend(loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


# -----------------------------
# Save / load (kept, but aligned)
# -----------------------------
def save_model_bundle(
    model: MultiModalVAE,
    history: Dict[str, list],
    metrics: Dict[str, float],
    filepath: Path
) -> None:
    bundle = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "model_config": {
            "wave_dim": model.wave_dim,
            "struct_dim": model.struct_dim,
            "latent_dim": model.latent_dim,
        }
    }
    torch.save(bundle, filepath)
    print(f"[Info] Saved model bundle to: {filepath}")


def load_model_bundle(filepath: Path, device: torch.device) -> Tuple[MultiModalVAE, Dict]:
    ckpt = torch.load(filepath, map_location=device)
    cfg = ckpt["model_config"]
    model = MultiModalVAE(
        wave_dim=cfg["wave_dim"],
        struct_dim=cfg["struct_dim"],
        latent_dim=cfg["latent_dim"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Info] Loaded model bundle from: {filepath}")
    return model, ckpt


# -----------------------------
# Main (training + evaluation)
# -----------------------------
if __name__ == "__main__":
    set_seed(SEED)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running on {device}")

    # 1) Load structured features (19) and z-score
    params = [
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

    struct_data = pd.read_csv(EVENT_TABLE_CSV, usecols=params)
    struct_data = zscore_dataframe(struct_data)
    struct_np = struct_data.values.astype(np.float32)

    # 2) Load waveforms
    with open(WAVE_PKL, "rb") as f:
        wave_dict = pickle.load(f)

    # Keep your expected key name; if your pkl uses another key, adjust here.
    wave_list = wave_dict.get("signal_list", None)
    if wave_list is None:
        raise KeyError("Expected key 'signal_list' in waveform pickle.")

    wave_lengths = np.array([len(w) for w in wave_list], dtype=int)
    target_length = int(np.quantile(wave_lengths, 0.95))
    print(f"[Info] Fixed waveform length (95th percentile): {target_length}")

    padded_wave, mask = standardize_per_event_and_pad(wave_list, target_length=target_length)

    # 3) Split
    X_train, X_val, s_train, s_val, m_train, m_val = train_test_split(
        padded_wave,
        struct_np,
        mask,
        test_size=0.2,
        random_state=SEED
    )

    # 4) Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    s_train_t = torch.tensor(s_train, dtype=torch.float32)
    s_val_t = torch.tensor(s_val, dtype=torch.float32)
    m_train_t = torch.tensor(m_train, dtype=torch.float32)
    m_val_t = torch.tensor(m_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, s_train_t, m_train_t)
    val_ds = TensorDataset(X_val_t, s_val_t, m_val_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) Model (latent_dim = 64)
    model = MultiModalVAE(
        wave_dim=X_train_t.shape[1],
        struct_dim=s_train_t.shape[1],
        latent_dim=LATENT_DIM
    ).to(device)

    # 6) Train (Adam, lr=1e-3, batch=128, up to 500, early stopping on val)
    model, history, train_metrics = train_vae(model, train_loader, val_loader, device)

    # Save best checkpoint (already saved during training)
    print(f"[Info] Best checkpoint: {CHECKPOINT_PATH}")

    # 7) Inference: latent mean is the final embedding (64-d)
    X_all_t = torch.tensor(padded_wave, dtype=torch.float32, device=device)
    s_all_t = torch.tensor(struct_np, dtype=torch.float32, device=device)
    z_mean_all = get_latent_mean_embedding(model, X_all_t, s_all_t)
    print(f"[Info] z_mean embedding shape: {z_mean_all.shape}")

    # 8) Reconstruction performance evaluation (Supplemental)
    val_metrics = calculate_reconstruction_metrics(
        model=model,
        wave_data=X_val_t,
        struct_data=s_val_t,
        mask_data=m_val_t,
        device=device
    )
    print("[Info] Reconstruction metrics (validation):")
    for k, v in val_metrics.items():
        print(f"  - {k}: {v:.6f}")

    # Optional: DTW/Cosine trend evaluation on a subset (kept, but not part of training)
    if _HAS_DTW:
        # Example: evaluate on up to 2000 samples to avoid huge runtime
        n_eval = min(2000, len(X_val_t))
        idx = np.random.default_rng(SEED).choice(len(X_val_t), size=n_eval, replace=False)
        wave_s = X_val_t[idx].to(device)
        struct_s = s_val_t[idx].to(device)
        mask_s = m_val_t[idx].to(device)

        with torch.no_grad():
            wave_recon, _, _, _ = model(wave_s, struct_s)

        dtws = []
        coss = []
        for i in range(n_eval):
            orig = wave_s[i].detach().cpu().numpy()
            rec = wave_recon[i].detach().cpu().numpy()
            ms = mask_s[i].detach().cpu().numpy()
            dtws.append(calculate_dtw_distance(orig, rec, ms))
            coss.append(calculate_cosine_similarity(orig, rec, ms))

        dtws = np.array([d for d in dtws if not np.isnan(d)])
        coss = np.array([c for c in coss if not np.isnan(c)])
        print("[Info] Waveform trend consistency (subset):")
        print(f"  - DTW (mean±std): {dtws.mean():.4f} ± {dtws.std():.4f} (lower is better)")
        print(f"  - Cosine (mean±std): {coss.mean():.4f} ± {coss.std():.4f} (higher is better)")

    # 9) Plotting (optional)
    plot_loss_curves(history)
    plot_wave_reconstruction(
        model=model,
        wave_data=X_val_t,
        struct_data=s_val_t,
        mask_data=m_val_t,
        device=device,
        num_samples=3,
        save_path=None
    )

    # 10) Save a bundled model + history + metrics (for later loading)
    all_metrics = {**train_metrics, **val_metrics}
    save_model_bundle(model, history, all_metrics, MODEL_BUNDLE_PATH)

    print("[Done] Training + evaluation finished.")

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, ckpt = load_model_bundle(MODEL_BUNDLE_PATH, device)
print(ckpt["metrics"])
