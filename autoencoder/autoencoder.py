
"""Autoencoder utilities for forecast time-series compression.

This module provides a small PyTorch autoencoder that compresses forecast
windows (PV generation, inflexible loads, and optional extra signals) into a
latent vector that can be used as part of RL state generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ArrayLike = Union[np.ndarray, Sequence[float]]


class _MLPAutoencoder(nn.Module):
    """Simple fully-connected autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if latent_dim >= input_dim:
            raise ValueError("latent_dim must be smaller than input_dim")

        act_layer = self._get_activation(activation)

        encoder_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), act_layer()])
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), act_layer()])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def _get_activation(name: str):
        name = name.lower().strip()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        if name not in activations:
            allowed = ", ".join(sorted(activations.keys()))
            raise ValueError(f"Unsupported activation '{name}'. Allowed: {allowed}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class AEConfig:
    """Configuration used to build and train an autoencoder."""

    input_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int, ...] = (128, 64)
    activation: str = "relu"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 64
    val_split: float = 0.1
    seed: int = None


class AE:
    """Forecast autoencoder wrapper used by state generation.

    Typical usage:
    1) Train once on historical/forecast windows:
       `ae.fit_from_forecasts(pv, inflexible_load, other_series=...)`
    2) At each environment step, compress a horizon window:
       `latent = ae.encode_state_features(pv_h, load_h, other_h)`
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        seed: int = None,
    ) -> None:
        self.config = AEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        #torch.manual_seed(seed)
        #np.random.seed(seed)

        self.model = _MLPAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(self.device)

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._is_fitted = False

    @staticmethod
    def _as_2d_array(x: ArrayLike, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")
        return arr

    @staticmethod
    def _rolling_windows(series: ArrayLike, horizon: int, stride: int = 1) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float32).reshape(-1)
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if arr.size < horizon:
            raise ValueError(
                f"Series length ({arr.size}) must be >= horizon ({horizon})."
            )

        starts = range(0, arr.size - horizon + 1, stride)
        windows = [arr[i : i + horizon] for i in starts]
        return np.asarray(windows, dtype=np.float32)

    @classmethod
    def build_single_series_matrix(
        cls,
        series: ArrayLike,
        horizon: Optional[int] = None,
        stride: int = 1,
    ) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim == 1:
            if horizon is None:
                raise ValueError("horizon is required when series is 1D.")
            return cls._rolling_windows(arr, horizon=horizon, stride=stride)
        if arr.ndim == 2:
            return arr.astype(np.float32)
        raise ValueError(f"series must be 1D or 2D, got shape {arr.shape}")

    def _fit_scaler(self, x: np.ndarray) -> None:
        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        self._std = np.where(std < 1e-8, 1.0, std)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler is not fitted. Train AE first.")
        return (x - self._mean) / self._std

    def fit(
        self,
        x: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        val_split: float = 0.1,
        verbose: bool = True,
        best_checkpoint_path: Optional[Union[str, Path]] = None,
        best_val_loss: float = np.inf,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder on a pre-built [N, D] matrix."""
        x = self._as_2d_array(x, "x")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.config.input_dim}, "
                f"got {x.shape[1]}."
            )
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1).")

        self._fit_scaler(x)
        x_norm = self._normalize(x).astype(np.float32)

        n = x_norm.shape[0]
        perm = np.random.permutation(n)
        x_norm = x_norm[perm]
        n_val = int(n * val_split)
        x_val = x_norm[:n_val] if n_val > 0 else None
        x_train = x_norm[n_val:]

        if x_train.shape[0] == 0:
            raise ValueError("No training samples left. Reduce val_split or add data.")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train)),
            batch_size=min(batch_size, x_train.shape[0]),
            shuffle=True,
            drop_last=False,
        )

        val_tensor = (
            torch.from_numpy(x_val).to(self.device) if x_val is not None else None
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path else None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_metric = float("inf")

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            n_train = 0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item()) * batch_x.size(0)
                n_train += batch_x.size(0)

            train_loss = train_loss / max(1, n_train)
            history["train_loss"].append(train_loss)

            if val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    recon_val = self.model(val_tensor)
                    val_loss = float(criterion(recon_val, val_tensor).item())
                self.model.train()
            else:
                val_loss = float("nan")
            history["val_loss"].append(val_loss)

            metric = val_loss if not np.isnan(val_loss) else train_loss
            if checkpoint_path is not None and metric < best_metric:
                best_metric = metric
                if metric < best_val_loss:
                    self.save(checkpoint_path)

            if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs):
                if np.isnan(val_loss):
                    print(f"[AE] epoch {epoch + 1:4d}/{epochs} - train_loss={train_loss:.6f}")
                else:
                    print(
                        f"[AE] epoch {epoch + 1:4d}/{epochs} - "
                        f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                    )
                if checkpoint_path is not None and metric <= best_metric:
                    print(f"[AE] best checkpoint updated: {checkpoint_path}")

        self._is_fitted = True
        return history

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode [N, D] input into [N, latent_dim]."""
        if not self._is_fitted:
            raise RuntimeError("AE must be trained (fit) before calling encode.")
        x = self._as_2d_array(x, "x")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.config.input_dim}, got {x.shape[1]}."
            )
        x_norm = self._normalize(x).astype(np.float32)
        with torch.no_grad():
            z = self.model.encoder(torch.from_numpy(x_norm).to(self.device))
        return z.cpu().numpy().astype(np.float32)

    def decode(self, z: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """Decode latent vectors [N, latent_dim] into reconstructed [N, D]."""
        if not self._is_fitted:
            raise RuntimeError("AE must be trained (fit) before calling decode.")
        z = self._as_2d_array(z, "z")
        if z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch. Expected {self.config.latent_dim}, got {z.shape[1]}."
            )
        with torch.no_grad():
            recon_norm = self.model.decoder(torch.from_numpy(z).to(self.device))
        recon = recon_norm.cpu().numpy().astype(np.float32)
        if denormalize:
            if self._mean is None or self._std is None:
                raise RuntimeError("Scaler is not available for denormalization.")
            recon = recon * self._std + self._mean
        return recon

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Convenience method: reconstruct input matrix x."""
        return self.decode(self.encode(x), denormalize=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save model + normalization metadata."""
        if self._mean is None or self._std is None:
            raise RuntimeError("Normalization parameters are missing.")
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "mean": self._mean.astype(np.float32),
            "std": self._std.astype(np.float32),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "AE":
        """Load model + normalization metadata."""
        payload = torch.load(
            str(path),
            map_location="cpu",
            # This checkpoint stores config/normalization numpy arrays in addition
            # to tensor weights, so it requires full (trusted) unpickling.
            weights_only=False,
        )
        cfg = payload["config"]
        model = cls(
            input_dim=int(cfg["input_dim"]),
            latent_dim=int(cfg["latent_dim"]),
            hidden_dims=tuple(cfg["hidden_dims"]),
            activation=str(cfg["activation"]),
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            device=device,
            seed=int(cfg["seed"]),
        )
        model.model.load_state_dict(payload["state_dict"])
        model._mean = np.asarray(payload["mean"], dtype=np.float32)
        model._std = np.asarray(payload["std"], dtype=np.float32)
        model._is_fitted = True
        model.model.eval()
        return model