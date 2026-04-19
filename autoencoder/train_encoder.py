"""Simple single-series autoencoder training pipeline.

Train one model per signal type (solar OR prices OR loads), e.g.:

python3 autoencoder/train_encoder.py \
  --signal-csv ev2gym/data/early_pv_scenario/test-pv.csv \
  --horizon 96 \
  --latent-dim 16 \
  --output-model autoencoder/models/solar_ae.pt
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import yaml

from autoencoder import AE

def _normalize_selector(selector: str) -> str:
    return selector.strip() if isinstance(selector, str) else "auto"


def _numeric_only_table(csv_path: Path) -> np.ndarray:
    table = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32, names=True)

    if table.size == 0:
        raise ValueError(f"No rows found in {csv_path}")

    if table.dtype.names is None:
        data = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        return np.asarray(data, dtype=np.float32)

    cols: List[np.ndarray] = []
    for name in table.dtype.names:
        col = np.asarray(table[name])
        if np.issubdtype(col.dtype, np.number):
            cols.append(col.astype(np.float32))

    if not cols:
        raise ValueError(f"No numeric columns found in {csv_path}")

    return np.column_stack(cols).astype(np.float32)

def _extend_series_timescale(series: np.ndarray, desired_timescale: int, dataset_timescale: int) -> np.ndarray:
    series = pd.Series(np.asarray(series, dtype=np.float32).reshape(-1))

    if desired_timescale > dataset_timescale:
        series = series.groupby(series.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        series = series.loc[series.index.repeat(dataset_timescale/desired_timescale)].reset_index(drop=True)

    # smooth data by taking the mean of every 5 rows
    series = series.rolling(window=60//desired_timescale, min_periods=1).mean()
    # use other type of smoothing
    series = series.ewm(span=60//desired_timescale, adjust=True).mean()
    
    return series.to_numpy(dtype=np.float32)


def _apply_state_like_pv_normalization(
    signal: np.ndarray,
    max_power_kw: float = 100.0,
) -> np.ndarray:
    """Match Transformer.normalize_pv_generation for AE pv training."""
    signal = -signal * max_power_kw

    return signal.astype(np.float32)


def _load_state_like_charge_price_series(
    csv_path: str,
    desired_timescale: int = 15,
    dataset_timescale: int = 60,
) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    data = pd.read_csv(path)
    if "Price (EUR/MWhe)" not in data.columns:
        raise ValueError(
            "Expected 'Price (EUR/MWhe)' column to match state price preprocessing."
        )

    prices = data["Price (EUR/MWhe)"].to_numpy(dtype=np.float32)

    if desired_timescale < dataset_timescale:
        if dataset_timescale % desired_timescale != 0:
            raise ValueError(
                f"Timescale mismatch: dataset_timescale={dataset_timescale} is not "
                f"divisible by desired_timescale={desired_timescale}."
            )
        repeat_factor = dataset_timescale // desired_timescale
        prices = np.repeat(prices, repeat_factor)
    elif desired_timescale > dataset_timescale:
        if desired_timescale % dataset_timescale != 0:
            raise ValueError(
                f"Timescale mismatch: desired_timescale={desired_timescale} is not "
                f"divisible by dataset_timescale={dataset_timescale}."
            )
        step = desired_timescale // dataset_timescale
        prices = prices[::step]

    # Match `load_electricity_prices`: charge price is negated and converted to EUR/kWh.
    return (-prices / 1000.0).astype(np.float32)


def _apply_state_like_inflexible_load_normalization(
    signal: np.ndarray,
    max_power_kw: float = 100.0,
) -> np.ndarray:
    """Match Transformer.normalize_inflexible_loads for AE load training."""

    # scale up the data to match the max_power of the transformers
    signal = signal * (max_power_kw /
                    signal.max()+0.0000001)
    # for each step
    for j in range(len(signal)):
        if signal[j] > max_power_kw:
            signal[j] = max_power_kw

        elif signal[j] < -max_power_kw:
            signal[j] = -max_power_kw

    return signal.astype(np.float32)


def _load_series(
    csv_path: str,
    column_selector: str = "auto",
) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    data = _numeric_only_table(path)
    selector = _normalize_selector(column_selector).lower()

    if selector == "all":
        return data.T.reshape(-1).astype(np.float32)

    if selector == "sample10sum":
        n_cols = data.shape[1]
        sample_size = min(10, n_cols)
        rng = np.random.default_rng()

        result_data = np.array([])
        for i in range(10):
            sampled_cols = rng.choice(n_cols, size=sample_size, replace=False)
            sampled_data = data[:, sampled_cols].sum(axis=1).astype(np.float32)
            result_data = np.concatenate([result_data, sampled_data])
        return result_data


    if selector.isdigit():
        idx = int(selector)
        if idx < 0 or idx >= data.shape[1]:
            raise ValueError(f"Column index {idx} out of range for {path}")

        column_data = data[:, idx]
        column_data = _extend_series_timescale(series=column_data, desired_timescale=15, dataset_timescale=60)
        return column_data

    raise ValueError(f"Column selector '{column_selector}' is unsupported without pandas. ")


def _save_training_representation(
    history: dict[str, list[float]],
    feature_matrix: np.ndarray,
    recon: np.ndarray,
    latents: np.ndarray,
    ae: AE,
    output_plot: Path,
) -> Path:
    # Decode explicitly from latent vectors (same path as reconstruct but shown separately).
    decoded = ae.decode(latents, denormalize=True)

    per_sample_mse = np.mean((recon - feature_matrix) ** 2, axis=1)
    sample_idx = int(np.argmax(per_sample_mse))

    fig, (ax_loss, ax_sig) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    ax_loss.plot(history.get("train_loss", []), label="train_loss", linewidth=2)
    if history.get("val_loss"):
        ax_loss.plot(history["val_loss"], label="val_loss", linewidth=2)
    ax_loss.set_title("Autoencoder training loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    x_axis = np.arange(feature_matrix.shape[1], dtype=int)
    ax_sig.plot(x_axis, feature_matrix[sample_idx], label="original", linewidth=2)
    ax_sig.plot(x_axis, recon[sample_idx], label="reconstructed", linewidth=2, linestyle="--")
    ax_sig.set_title(f"Window comparison (sample index {sample_idx}, highest reconstruction error)")
    ax_sig.set_xlabel("Step in horizon")
    ax_sig.set_ylabel("Signal value")
    ax_sig.grid(alpha=0.3)
    ax_sig.legend()

    plot_path = output_plot
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def main_pipeline(signal_csv: str, signal_column: str, horizon: int, latent_dim: int, hidden_dims: list[int], activation: str, learning_rate: float, weight_decay: float, epochs: int, batch_size: int, val_split: float, seed: int, device: str, output_model: str, output_plot: str, output_latents: str, data_type: str, env_config_path: str = "") -> None:
    best_val_loss = np.inf
    best_train_loss = np.inf

    for run_idx in range(1):
        signal = _load_series(
                signal_csv,
                column_selector=signal_column,
            )
        #print("original signal mean: ", signal.mean())
        #print("original signal std: ", signal.std())

        if data_type == "prices":
            signal = _load_state_like_charge_price_series(
                signal_csv,
                desired_timescale=15,
                dataset_timescale=60,
            )
        elif data_type == "loads":
            signal = _apply_state_like_inflexible_load_normalization(signal)
            #print("normalized signal mean: ", signal.mean())
            #print("normalized signal std: ", signal.std())

        elif data_type == "solar":
            signal = _apply_state_like_pv_normalization(signal)
        
        feature_matrix = AE.build_single_series_matrix(series=signal, horizon=horizon, stride=1)

        input_dim = feature_matrix.shape[1]
        if latent_dim >= input_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be < input_dim ({input_dim}). "
                "Reduce latent size or increase number of signals/horizon."
            )

        ae = AE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )

        out_model = Path(output_model)
        out_model.parent.mkdir(parents=True, exist_ok=True)

        #print(f"[AE train] feature_matrix shape={feature_matrix.shape}")

        print(f"[TRAIN SNOOP] Matrix max: {feature_matrix.max():.2f}, min: {feature_matrix.min():.2f}, mean: {feature_matrix.mean():.2f}")


        history = ae.fit(
            x=feature_matrix,
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            verbose=False,
            best_checkpoint_path=out_model,
        )

        # Reload to ensure reported metrics/latents come from the saved best checkpoint.
        ae = AE.load(out_model, device=device)
        latents = ae.encode(feature_matrix)
        recon = ae.reconstruct(feature_matrix)
        mse = float(np.mean((recon - feature_matrix) ** 2))
        

        if output_latents:
            out_latents = Path(output_latents)
            out_latents.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_latents), latents)
            print(f"[AE train] Saved latents to: {out_latents} | shape={latents.shape}")

        final_train = history["train_loss"][-1] if history["train_loss"] else float("nan")
        final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")

        if final_val < best_val_loss:
            print(f"Best val loss updated: {final_val:.6f}")
            best_val_loss = final_val
            best_train_loss = final_train
            plot_path = _save_training_representation(
                history=history,
                feature_matrix=feature_matrix,
                recon=recon,
                latents=latents,
                ae=ae,
                output_plot=output_plot,
            )

        print("===================================================")
        print(f"[AE train] Model saved to: {out_model}")
        print(f"[AE train] Signal CSV: {signal_csv}")
        print(f"[AE train] Latent shape: {latents.shape}")
        print(f"[AE train] Final train loss: {final_train:.6f}")
        print(f"[AE train] Final val loss:   {final_val:.6f}")
        print(f"[AE train] Reconstruction MSE (denormalized): {mse:.6f}")
        print(f"[AE train] Training representation saved to: {plot_path}")
        print("===================================================")
    
    with open(f"autoencoder/log_N.txt", "a") as f:
        f.write(f"best value loss: {best_val_loss:.6f}\nbest train loss: {best_train_loss:.6f}\nsaved to: {out_model}\n\n")



def train_for_solar(latent_dim: int):
    main_pipeline(signal_csv="ev2gym/data/pv_netherlands.csv", #"ev2gym/data/test/pv_encode_test.csv",
        signal_column="2", 
        horizon=96, 
        latent_dim=latent_dim,
        hidden_dims=[64, 32], 
        activation="relu", 
        learning_rate=1e-3, 
        weight_decay=0.0, 
        epochs=100, 
        batch_size=64, 
        val_split=0.1, 
        seed=random.randint(0, 1000000), #42, 
        device=None, 
        output_model=f"autoencoder/models/N_solar_ae_to{latent_dim}dim.pt", 
        output_plot=f"autoencoder/plots/N_solar_ae_to{latent_dim}dim.training_plot.png",
        output_latents="",
        data_type="solar",
        env_config_path="")


def train_for_prices(latent_dim: int):
    main_pipeline(signal_csv="ev2gym/data/Netherlands_day-ahead-2015-2024.csv",
        signal_column="3",
        horizon=96,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        learning_rate=1e-3,
        weight_decay=0.0,
        epochs=100,
        batch_size=64,
        val_split=0.1,
        seed=random.randint(0, 1000000), #42, 
        device=None, 
        output_model=f"autoencoder/models/N_prices_ae_to{latent_dim}dim.pt", 
        output_plot=f"autoencoder/plots/N_prices_ae_to{latent_dim}dim.training_plot.png",
        output_latents="",
        data_type="prices",
        env_config_path="")


def train_for_loads(latent_dim: int):
    main_pipeline(signal_csv="ev2gym/data/residential_loads.csv",
        signal_column="sample10sum",
        horizon=96,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        learning_rate=1e-3,
        weight_decay=0.0,
        epochs=100,
        batch_size=64,
        val_split=0.1,
        seed=random.randint(0, 1000000), #42, 
        device=None, 
        output_model=f"autoencoder/models/N_loads_ae_to{latent_dim}dim.pt", 
        output_plot=f"autoencoder/plots/N_loads_ae_to{latent_dim}dim.training_plot.png",
        output_latents="",
        data_type="loads",
        env_config_path="ev2gym/example_config_files/V2GProfitPlusLoads.yaml")

if __name__ == "__main__":
    train_for_loads(latent_dim=2)
    train_for_loads(latent_dim=4)
    train_for_loads(latent_dim=8)
    train_for_loads(latent_dim=16)
    train_for_prices(latent_dim=2)
    train_for_prices(latent_dim=4)
    train_for_prices(latent_dim=8)
    train_for_prices(latent_dim=16)
    train_for_solar(latent_dim=2)
    train_for_solar(latent_dim=4)
    train_for_solar(latent_dim=8)
    train_for_solar(latent_dim=16)

