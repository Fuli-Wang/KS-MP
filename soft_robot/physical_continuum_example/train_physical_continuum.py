#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a learned body schema for the physical continuum-robot example.

The platform-specific model maps four motor commands to the camera-measured
planar tip position:

    q = [ID1, ID4, ID5, ID6]
    x = [x_mm, y_mm]

The script aligns a motor-command schedule with ZED measurements by
``sample_id``, trains a compact differentiable MLP, evaluates Cartesian
prediction errors, and exports a checkpoint that supports automatic-
differentiation Jacobian evaluation.

Example
-------
python train_physical_continuum.py \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Learned body schema
# ---------------------------------------------------------------------------

class PhysicalContinuumPosNet(nn.Module):
    """
    Differentiable body schema mapping motor commands to planar tip position.

    Training uses normalised coordinates. ``predict()`` and ``jacobian()``
    operate in the original motor-command and millimetre units.
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 2,
        widths: Sequence[int] = (64, 64, 32),
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        previous = in_dim
        for width in widths:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.SiLU(),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, out_dim))
        self.network = nn.Sequential(*layers)

        self.register_buffer("q_mean", torch.zeros(in_dim))
        self.register_buffer("q_std", torch.ones(in_dim))
        self.register_buffer("x_mean", torch.zeros(out_dim))
        self.register_buffer("x_std", torch.ones(out_dim))

    def forward(self, q_norm: torch.Tensor) -> torch.Tensor:
        return self.network(q_norm)

    def set_normalisation(
        self,
        q_mean: np.ndarray,
        q_std: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> None:
        self.q_mean.copy_(torch.as_tensor(q_mean, dtype=torch.float32))
        self.q_std.copy_(torch.as_tensor(q_std, dtype=torch.float32))
        self.x_mean.copy_(torch.as_tensor(x_mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(x_std, dtype=torch.float32))

    @torch.no_grad()
    def predict(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = q.to(self.q_mean.device, dtype=torch.float32)
        q_norm = (q - self.q_mean) / self.q_std
        x_norm = self.forward(q_norm)
        return self.x_mean + self.x_std * x_norm

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """
        Return J = d[x_mm, y_mm] / d[ID1, ID4, ID5, ID6]
        with shape [batch, 2, 4].
        """
        if q.ndim == 1:
            q = q.unsqueeze(0)

        q = q.to(self.q_mean.device, dtype=torch.float32)
        q_norm = ((q - self.q_mean) / self.q_std).detach()
        q_norm.requires_grad_(True)

        x_norm = self.forward(q_norm)
        jac_rows = []

        for axis in range(x_norm.shape[1]):
            grad_outputs = torch.zeros_like(x_norm)
            grad_outputs[:, axis] = 1.0
            grad = torch.autograd.grad(
                outputs=x_norm,
                inputs=q_norm,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )[0]
            jac_rows.append(grad.unsqueeze(1))

        jac_norm = torch.cat(jac_rows, dim=1)
        return (
            self.x_std.view(1, -1, 1)
            * jac_norm
            / self.q_std.view(1, 1, -1)
        )


# ---------------------------------------------------------------------------
# Data loading and alignment
# ---------------------------------------------------------------------------

REQUIRED_SCHEDULE_COLUMNS = ["sample_id", "ID1", "ID4", "ID5", "ID6"]
REQUIRED_ZED_COLUMNS = ["sample_id", "x_mm", "y_mm"]


def _normalise_header(name: str) -> str:
    return name.strip().lstrip("\ufeff")


def read_delimited_table(path: Path) -> List[Dict[str, str]]:
    """
    Read CSV, TSV, or whitespace-delimited TXT with a header row.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    nonempty = [line for line in text.splitlines() if line.strip()]
    if not nonempty:
        raise ValueError(f"File is empty: {path}")

    first = nonempty[0]
    if "," in first:
        delimiter = ","
    elif "\t" in first:
        delimiter = "\t"
    else:
        delimiter = None

    rows: List[Dict[str, str]] = []

    if delimiter is not None:
        reader = csv.DictReader(nonempty, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        reader.fieldnames = [_normalise_header(x) for x in reader.fieldnames]
        for row in reader:
            rows.append(
                {
                    _normalise_header(str(k)): str(v).strip()
                    for k, v in row.items()
                    if k is not None
                }
            )
    else:
        header = [_normalise_header(x) for x in first.split()]
        for line in nonempty[1:]:
            values = line.split()
            if len(values) < len(header):
                continue
            rows.append(dict(zip(header, values[: len(header)])))

    return rows


def require_columns(
    rows: List[Dict[str, str]],
    required: Sequence[str],
    source_name: str,
) -> None:
    if not rows:
        raise ValueError(f"No rows found in {source_name}")
    available = set(rows[0].keys())
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(
            f"{source_name} is missing columns {missing}. "
            f"Available columns: {sorted(available)}"
        )


def merge_schedule_and_zed(
    schedule_rows: List[Dict[str, str]],
    zed_rows: List[Dict[str, str]],
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    require_columns(schedule_rows, REQUIRED_SCHEDULE_COLUMNS, "schedule")
    require_columns(zed_rows, REQUIRED_ZED_COLUMNS, "ZED data")

    schedule_by_id = {
        int(float(row["sample_id"])): row for row in schedule_rows
    }
    zed_by_id = {
        int(float(row["sample_id"])): row for row in zed_rows
    }

    common_ids = sorted(set(schedule_by_id) & set(zed_by_id))
    if max_samples is not None:
        common_ids = common_ids[:max_samples]

    if not common_ids:
        raise ValueError("No common sample_id values were found.")

    merged: List[Dict[str, object]] = []
    q_list: List[List[float]] = []
    x_list: List[List[float]] = []

    for sample_id in common_ids:
        s = schedule_by_id[sample_id]
        z = zed_by_id[sample_id]

        q = [
            float(s["ID1"]),
            float(s["ID4"]),
            float(s["ID5"]),
            float(s["ID6"]),
        ]
        x = [float(z["x_mm"]), float(z["y_mm"])]

        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(x)):
            continue

        q_list.append(q)
        x_list.append(x)
        merged.append(
            {
                "sample_id": sample_id,
                "ID1": q[0],
                "ID4": q[1],
                "ID5": q[2],
                "ID6": q[3],
                "x_mm": x[0],
                "y_mm": x[1],
            }
        )

    if len(merged) < 30:
        raise ValueError(
            f"Only {len(merged)} valid aligned samples were found; "
            "at least 30 are recommended."
        )

    return (
        np.asarray(q_list, dtype=np.float32),
        np.asarray(x_list, dtype=np.float32),
        merged,
    )


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def grid_aware_split(
    q: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out interleaved command-grid points for validation and testing.

    This split is specific to the sampled two-dimensional command grid used
    by the physical platform. A deterministic random split is used as a
    fallback for non-grid datasets.
    """
    id1_values = sorted(np.unique(q[:, 0]).tolist())
    id5_values = sorted(np.unique(q[:, 2]).tolist())

    id1_rank = {value: i for i, value in enumerate(id1_values)}
    id5_rank = {value: i for i, value in enumerate(id5_values)}

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for idx, row in enumerate(q):
        i = id1_rank[float(row[0])]
        j = id5_rank[float(row[2])]

        # Interleaved held-out interior points.
        if i % 4 == 2 and j % 4 == 2:
            test_idx.append(idx)
        elif i % 4 == 0 and j % 4 == 0:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

    # Fallback for unusual/non-grid datasets.
    if len(val_idx) < 8 or len(test_idx) < 8:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(q))
        n_test = max(1, round(0.15 * len(q)))
        n_val = max(1, round(0.15 * len(q)))
        test_idx = perm[:n_test].tolist()
        val_idx = perm[n_test : n_test + n_val].tolist()
        train_idx = perm[n_test + n_val :].tolist()

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

class NormalisedDataset(Dataset):
    def __init__(
        self,
        q: np.ndarray,
        x: np.ndarray,
        q_mean: np.ndarray,
        q_std: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> None:
        self.q = torch.as_tensor(
            (q - q_mean) / q_std, dtype=torch.float32
        )
        self.x = torch.as_tensor(
            (x - x_mean) / x_std, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, index: int):
        return self.q[index], self.x[index]


@dataclass
class TrainingConfig:
    seed: int = 42
    widths: Tuple[int, int, int] = (64, 64, 32)
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 3000
    patience: int = 250
    min_delta: float = 1.0e-6


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total = 0.0
    count = 0

    for q_batch, x_batch in loader:
        q_batch = q_batch.to(device)
        x_batch = x_batch.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(q_batch)
        loss = nn.functional.smooth_l1_loss(
            pred, x_batch, beta=0.5
        )

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_n = q_batch.shape[0]
        total += float(loss.detach().cpu()) * batch_n
        count += batch_n

    return total / max(1, count)


@torch.no_grad()
def predict_numpy(
    model: PhysicalContinuumPosNet,
    q: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.as_tensor(q, dtype=torch.float32, device=device)
    return model.predict(tensor).cpu().numpy()


def metrics_2d(
    truth: np.ndarray,
    pred: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    error = pred - truth
    euclidean = np.linalg.norm(error, axis=1)
    axis_rmse = np.sqrt(np.mean(error**2, axis=0))

    return {
        f"{prefix}_n": int(len(truth)),
        f"{prefix}_rmse_x_mm": float(axis_rmse[0]),
        f"{prefix}_rmse_y_mm": float(axis_rmse[1]),
        f"{prefix}_rmse_euclidean_mm": float(
            np.sqrt(np.mean(euclidean**2))
        ),
        f"{prefix}_mean_euclidean_mm": float(np.mean(euclidean)),
        f"{prefix}_median_euclidean_mm": float(np.median(euclidean)),
        f"{prefix}_p95_euclidean_mm": float(
            np.percentile(euclidean, 95)
        ),
        f"{prefix}_max_euclidean_mm": float(np.max(euclidean)),
    }


def save_table(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zed",
        type=Path,
        default=Path("example_data/zed_measurements.csv"),
        help="ZED CSV or TXT containing sample_id, x_mm, y_mm.",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("example_data/sampling_schedule.csv"),
        help="Motor command schedule CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of aligned samples. Default: use all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=250)
    args = parser.parse_args()

    config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    seed_everything(config.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    schedule_rows = read_delimited_table(args.schedule)
    zed_rows = read_delimited_table(args.zed)

    q, x, merged_rows = merge_schedule_and_zed(
        schedule_rows=schedule_rows,
        zed_rows=zed_rows,
        max_samples=args.max_samples,
    )

    train_idx, val_idx, test_idx = grid_aware_split(q, config.seed)

    q_train, x_train = q[train_idx], x[train_idx]
    q_val, x_val = q[val_idx], x[val_idx]
    q_test, x_test = q[test_idx], x[test_idx]

    q_mean = q_train.mean(axis=0)
    q_std = q_train.std(axis=0)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    q_std = np.where(q_std < 1e-8, 1.0, q_std).astype(np.float32)
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)

    train_ds = NormalisedDataset(
        q_train, x_train, q_mean, q_std, x_mean, x_std
    )
    val_ds = NormalisedDataset(
        q_val, x_val, q_mean, q_std, x_mean, x_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(config.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(128, len(val_ds)),
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PhysicalContinuumPosNet(
        in_dim=4,
        out_dim=2,
        widths=config.widths,
    ).to(device)
    model.set_normalisation(q_mean, q_std, x_mean, x_std)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=60,
        min_lr=1e-6,
    )

    best_val = math.inf
    best_state = None
    stale_epochs = 0
    history_rows: List[Dict[str, object]] = []

    print(
        f"Aligned samples: {len(q)} | "
        f"train/val/test = {len(train_idx)}/"
        f"{len(val_idx)}/{len(test_idx)}"
    )
    print(f"Device: {device}")
    print("Input order: [ID1, ID4, ID5, ID6]")
    print("Output order: [x_mm, y_mm]")

    for epoch in range(1, config.max_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, val_loader, device, optimizer=None
            )
        scheduler.step(val_loss)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss_normalised": train_loss,
                "val_loss_normalised": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | train={train_loss:.6e} | "
                f"val={val_loss:.6e} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    pred_train = predict_numpy(model, q_train, device)
    pred_val = predict_numpy(model, q_val, device)
    pred_test = predict_numpy(model, q_test, device)
    pred_all = predict_numpy(model, q, device)

    metrics: Dict[str, object] = {
        "num_samples_total": int(len(q)),
        "num_train": int(len(train_idx)),
        "num_validation": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "input_order": ["ID1", "ID4", "ID5", "ID6"],
        "output_order": ["x_mm", "y_mm"],
        "best_validation_loss_normalised": float(best_val),
        **metrics_2d(x_train, pred_train, "train"),
        **metrics_2d(x_val, pred_val, "validation"),
        **metrics_2d(x_test, pred_test, "test"),
    }

    checkpoint = {
        "model_class": "PhysicalContinuumPosNet",
        "model_version": 1,
        "in_dim": 4,
        "out_dim": 2,
        "widths": list(config.widths),
        "input_order": ["ID1", "ID4", "ID5", "ID6"],
        "output_order": ["x_mm", "y_mm"],
        "state_dict": model.state_dict(),
        "training_config": asdict(config),
        "metrics": metrics,
    }

    torch.save(
        checkpoint,
        args.output_dir / "physical_continuum_body_schema.pt",
    )

    with (
        args.output_dir / "metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_table(
        args.output_dir / "training_history.csv",
        history_rows,
    )
    save_table(
        args.output_dir / "merged_dataset.csv",
        merged_rows,
    )

    split_labels = np.full(len(q), "train", dtype=object)
    split_labels[val_idx] = "validation"
    split_labels[test_idx] = "test"

    prediction_rows = []
    for i, row in enumerate(merged_rows):
        error = pred_all[i] - x[i]
        prediction_rows.append(
            {
                **row,
                "split": split_labels[i],
                "pred_x_mm": float(pred_all[i, 0]),
                "pred_y_mm": float(pred_all[i, 1]),
                "error_x_mm": float(error[0]),
                "error_y_mm": float(error[1]),
                "error_euclidean_mm": float(np.linalg.norm(error)),
            }
        )

    save_table(
        args.output_dir / "predictions_all.csv",
        prediction_rows,
    )

    print("\nBody-schema training complete.")
    print(
        f"Test RMSE (Euclidean): "
        f"{metrics['test_rmse_euclidean_mm']:.3f} mm"
    )
    print(
        f"Test p95 error: "
        f"{metrics['test_p95_euclidean_mm']:.3f} mm"
    )
    print(
        f"Checkpoint: "
        f"{(args.output_dir / 'physical_fk_4to2_best.pt').resolve()}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
