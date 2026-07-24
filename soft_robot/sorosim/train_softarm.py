#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate a learned soft-arm body schema.

The model maps internal soft-arm coordinates to Cartesian tip position and
provides an analytical Jacobian through automatic differentiation. The script
supports deterministic data splitting, physical-space error metrics, coverage
diagnostics, Jacobian checks, and checkpoint export.

Expected dataset:
- Q: internal coordinates, shape [N, n_dof]
- X: Cartesian tip positions in mm, shape [N, 3]
"""

import argparse
import copy
import csv
import json
import os
import random
import time
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Learned body-schema model
# -----------------------------------------------------------------------------
class SoftArmPosNet(nn.Module):
    """
    MLP body schema mapping internal coordinates q to Cartesian tip position x.

    Training uses normalised coordinates, while predict() and jacobian()
    operate in the original physical units.
    """

    def __init__(self, in_dim: int = 18, out_dim: int = 3, width: int = 256, depth: int = 4):
        super().__init__()

        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

        # Normalisation statistics are stored with the model state.
        self.register_buffer("q_mean", torch.zeros(in_dim))
        self.register_buffer("q_std", torch.ones(in_dim))
        self.register_buffer("x_mean", torch.zeros(out_dim))
        self.register_buffer("x_std", torch.ones(out_dim))

    def forward(self, q_norm: torch.Tensor) -> torch.Tensor:
        h = self.backbone(q_norm)
        return self.head(h)

    def set_normalization(self, q_mean: np.ndarray, q_std: np.ndarray,
                          x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.q_mean.copy_(torch.from_numpy(q_mean).float())
        self.q_std.copy_(torch.from_numpy(q_std).float())
        self.x_mean.copy_(torch.from_numpy(x_mean).float())
        self.x_std.copy_(torch.from_numpy(x_std).float())

    @torch.no_grad()
    def predict(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [B, in_dim] in original data units.
        return x: [B, 3] in physical units, e.g. mm.
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(self.q_mean.device)
        q_norm = (q - self.q_mean) / self.q_std
        x_norm = self.forward(q_norm)
        return self.x_mean + self.x_std * x_norm

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [B, in_dim] in original data units.
        return J = dx/dq: [B, 3, in_dim] in physical units / q_unit.
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        device = self.q_mean.device
        q = q.to(device)

        q_norm = (q - self.q_mean) / self.q_std
        q_norm.requires_grad_(True)

        x_norm = self.forward(q_norm)  # [B, 3]
        _, out_dim = x_norm.shape

        jac_norm = []
        for i in range(out_dim):
            grad_out = torch.zeros_like(x_norm)
            grad_out[:, i] = 1.0
            grad_q_norm = torch.autograd.grad(
                outputs=x_norm,
                inputs=q_norm,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0]
            jac_norm.append(grad_q_norm.unsqueeze(1))

        J_norm = torch.cat(jac_norm, dim=1)  # [B, 3, in_dim]

        # x = x_mean + x_std * x_norm, q_norm = (q - q_mean) / q_std
        sigma_x = self.x_std.view(1, -1, 1)
        inv_sigma_q = (1.0 / self.q_std).view(1, 1, -1)
        return sigma_x * J_norm * inv_sigma_q


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class QXDataset(Dataset):
    def __init__(self, Q: np.ndarray, X: np.ndarray,
                 q_mean: np.ndarray, q_std: np.ndarray,
                 x_mean: np.ndarray, x_std: np.ndarray):
        self.Q = torch.from_numpy(Q).float()
        self.X = torch.from_numpy(X).float()
        self.q_mean = torch.from_numpy(q_mean).float()
        self.q_std = torch.from_numpy(q_std).float()
        self.x_mean = torch.from_numpy(x_mean).float()
        self.x_std = torch.from_numpy(x_std).float()

    def __len__(self) -> int:
        return self.Q.shape[0]

    def __getitem__(self, idx: int):
        q = self.Q[idx]
        x = self.X[idx]
        q_norm = (q - self.q_mean) / self.q_std
        x_norm = (x - self.x_mean) / self.x_std
        return q_norm, x_norm


# -----------------------------------------------------------------------------
# Training and validation
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for q_norm, x_norm in loader:
        q_norm = q_norm.to(device)
        x_norm = x_norm.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(q_norm)
        loss = loss_fn(pred, x_norm)
        loss.backward()
        optimizer.step()

        bs = q_norm.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for q_norm, x_norm in loader:
        q_norm = q_norm.to(device)
        x_norm = x_norm.to(device)
        pred = model(q_norm)
        loss = loss_fn(pred, x_norm)
        bs = q_norm.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


# -----------------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_numpy(model: SoftArmPosNet, Q: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    Q_t = torch.from_numpy(Q).float()
    preds = []
    for i in range(0, len(Q_t), batch_size):
        q_batch = Q_t[i:i + batch_size].to(device)
        preds.append(model.predict(q_batch).detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def physical_error_metrics(X_pred: np.ndarray, X_true: np.ndarray, prefix: str = "test") -> Dict[str, float]:
    err = X_pred - X_true
    per_axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    euclidean_err = np.linalg.norm(err, axis=1)

    return {
        f"{prefix}_rmse_x_mm": float(per_axis_rmse[0]),
        f"{prefix}_rmse_y_mm": float(per_axis_rmse[1]),
        f"{prefix}_rmse_z_mm": float(per_axis_rmse[2]),
        f"{prefix}_rmse_euclidean_mm": float(np.sqrt(np.mean(euclidean_err ** 2))),
        f"{prefix}_mean_euclidean_mm": float(np.mean(euclidean_err)),
        f"{prefix}_median_euclidean_mm": float(np.median(euclidean_err)),
        f"{prefix}_p95_euclidean_mm": float(np.percentile(euclidean_err, 95)),
        f"{prefix}_max_euclidean_mm": float(np.max(euclidean_err)),
    }


def nearest_neighbour_coverage_metrics(Q_train: np.ndarray, Q_eval: np.ndarray,
                                       X_pred: np.ndarray, X_true: np.ndarray,
                                       q_mean: np.ndarray, q_std: np.ndarray,
                                       prefix: str = "test") -> Dict[str, float]:
    """
    Measure nearest-neighbour coverage in normalised coordinate space and its
    correlation with Cartesian prediction error.
    """
    Q_train_norm = (Q_train - q_mean) / q_std
    Q_eval_norm = (Q_eval - q_mean) / q_std
    tree = cKDTree(Q_train_norm)
    nn_dist, _ = tree.query(Q_eval_norm, k=1)

    euclidean_err = np.linalg.norm(X_pred - X_true, axis=1)
    if np.std(nn_dist) > 1e-12 and np.std(euclidean_err) > 1e-12:
        corr = float(np.corrcoef(nn_dist, euclidean_err)[0, 1])
    else:
        corr = float("nan")

    return {
        f"{prefix}_nn_dist_mean_norm_q": float(np.mean(nn_dist)),
        f"{prefix}_nn_dist_median_norm_q": float(np.median(nn_dist)),
        f"{prefix}_nn_dist_p95_norm_q": float(np.percentile(nn_dist, 95)),
        f"{prefix}_nn_dist_max_norm_q": float(np.max(nn_dist)),
        f"{prefix}_err_nn_dist_corr": corr,
    }


def jacobian_condition_metrics(model: SoftArmPosNet, Q_eval: np.ndarray, device,
                               num_points: int = 200, seed: int = 42,
                               prefix: str = "test") -> Dict[str, float]:
    """Distribution of condition numbers of learned J = dx/dq over sampled queries."""
    if num_points <= 0 or len(Q_eval) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = min(num_points, len(Q_eval))
    idx = rng.choice(len(Q_eval), size=n, replace=False)
    Qs = torch.from_numpy(Q_eval[idx]).float().to(device)

    model.eval()
    J = model.jacobian(Qs).detach().cpu().numpy()  # [n, 3, in_dim]

    conds = []
    sigma_min = []
    sigma_max = []
    for Ji in J:
        s = np.linalg.svd(Ji, compute_uv=False)
        smax = float(np.max(s))
        smin = float(np.min(s))
        sigma_max.append(smax)
        sigma_min.append(smin)
        conds.append(float(smax / max(smin, 1e-12)))

    conds = np.asarray(conds)
    sigma_min = np.asarray(sigma_min)
    sigma_max = np.asarray(sigma_max)

    return {
        f"{prefix}_jacobian_cond_mean": float(np.mean(conds)),
        f"{prefix}_jacobian_cond_median": float(np.median(conds)),
        f"{prefix}_jacobian_cond_p95": float(np.percentile(conds, 95)),
        f"{prefix}_jacobian_cond_max": float(np.max(conds)),
        f"{prefix}_jacobian_sigma_min_mean": float(np.mean(sigma_min)),
        f"{prefix}_jacobian_sigma_max_mean": float(np.mean(sigma_max)),
    }


def finite_difference_jacobian(model: SoftArmPosNet, q: np.ndarray, device,
                               eps_scale: float = 1e-3) -> np.ndarray:
    """
    Central finite-difference Jacobian for one q.
    Step size is eps_scale * max(q_std_j, 1), so dimensions with tiny std do not collapse.
    """
    q = q.astype(np.float32).copy()
    in_dim = q.shape[0]
    J_fd = np.zeros((3, in_dim), dtype=np.float64)

    q_std = model.q_std.detach().cpu().numpy()
    steps = eps_scale * np.maximum(np.abs(q_std), 1.0)

    Q_batch = []
    for j in range(in_dim):
        qp = q.copy(); qm = q.copy()
        qp[j] += steps[j]
        qm[j] -= steps[j]
        Q_batch.append(qp)
        Q_batch.append(qm)

    Q_batch = np.stack(Q_batch, axis=0)
    X_batch = predict_numpy(model, Q_batch, device, batch_size=len(Q_batch))

    for j in range(in_dim):
        xp = X_batch[2 * j]
        xm = X_batch[2 * j + 1]
        J_fd[:, j] = (xp - xm) / (2.0 * steps[j])

    return J_fd


def jacobian_finite_difference_metrics(model: SoftArmPosNet, Q_eval: np.ndarray, device,
                                       num_points: int = 50, seed: int = 42,
                                       eps_scale: float = 1e-3,
                                       prefix: str = "test") -> Dict[str, float]:
    """
    Compare automatic-differentiation and finite-difference Jacobians.

    This checks numerical consistency within the learned model; it is not a
    comparison with a physical ground-truth Jacobian.
    """
    if num_points <= 0 or len(Q_eval) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = min(num_points, len(Q_eval))
    idx = rng.choice(len(Q_eval), size=n, replace=False)

    abs_fro = []
    rel_fro = []
    cos_vals = []

    model.eval()
    for k in idx:
        q = Q_eval[k]
        q_t = torch.from_numpy(q).float().unsqueeze(0).to(device)
        J_ad = model.jacobian(q_t)[0].detach().cpu().numpy().astype(np.float64)
        J_fd = finite_difference_jacobian(model, q, device, eps_scale=eps_scale)

        diff = J_ad - J_fd
        abs_val = np.linalg.norm(diff, ord="fro")
        denom = max(np.linalg.norm(J_fd, ord="fro"), 1e-12)
        rel_val = abs_val / denom

        ad_flat = J_ad.reshape(-1)
        fd_flat = J_fd.reshape(-1)
        cos = float(np.dot(ad_flat, fd_flat) / max(np.linalg.norm(ad_flat) * np.linalg.norm(fd_flat), 1e-12))

        abs_fro.append(abs_val)
        rel_fro.append(rel_val)
        cos_vals.append(cos)

    abs_fro = np.asarray(abs_fro)
    rel_fro = np.asarray(rel_fro)
    cos_vals = np.asarray(cos_vals)

    return {
        f"{prefix}_jacobian_fd_abs_fro_mean": float(np.mean(abs_fro)),
        f"{prefix}_jacobian_fd_abs_fro_p95": float(np.percentile(abs_fro, 95)),
        f"{prefix}_jacobian_fd_rel_fro_mean": float(np.mean(rel_fro)),
        f"{prefix}_jacobian_fd_rel_fro_p95": float(np.percentile(rel_fro, 95)),
        f"{prefix}_jacobian_fd_cosine_mean": float(np.mean(cos_vals)),
        f"{prefix}_jacobian_fd_cosine_min": float(np.min(cos_vals)),
    }


# -----------------------------------------------------------------------------
# Input and output helpers
# -----------------------------------------------------------------------------
def load_qx_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(path)
    if "Q" not in data or "X" not in data:
        raise KeyError(f"{path} must contain variables 'Q' and 'X'. Found keys: {list(data.keys())}")
    Q = np.asarray(data["Q"], dtype=np.float32)
    X = np.asarray(data["X"], dtype=np.float32)
    if Q.ndim != 2 or X.ndim != 2:
        raise ValueError(f"Expected Q and X to be 2D arrays, got Q{Q.shape}, X{X.shape}")
    if Q.shape[0] != X.shape[0]:
        raise ValueError(f"Q and X must have same number of rows, got {Q.shape[0]} and {X.shape[0]}")
    return Q, X


def save_metrics_csv(metrics: Dict[str, float], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def print_metric_block(title: str, metrics: Dict[str, float], keys=None) -> None:
    print("\n" + title)
    print("-" * len(title))
    if keys is None:
        keys = list(metrics.keys())
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"{k}: {v:.6g}")
            else:
                print(f"{k}: {v}")


# -----------------------------------------------------------------------------
# Training workflow
# -----------------------------------------------------------------------------
def main(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    print(f"Loading training/validation/test data from {args.mat_path}")
    Q, X = load_qx_mat(args.mat_path)
    N, in_dim = Q.shape
    out_dim = X.shape[1]
    print(f"Dataset size: {N}, in_dim: {in_dim}, out_dim: {out_dim}")

    if out_dim != 3:
        print(f"Warning: expected X to have 3 columns, but got out_dim={out_dim}.")

    # Split train/val/test
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_train = int(N * args.train_ratio)
    n_val = int(N * args.val_ratio)
    n_test = N - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Invalid split: train={n_train}, val={n_val}, test={n_test}. Adjust ratios or dataset size.")

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    Q_train, X_train = Q[idx_train], X[idx_train]
    Q_val, X_val = Q[idx_val], X[idx_val]
    Q_test, X_test = Q[idx_test], X[idx_test]

    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    # Compute normalisation statistics from the training split by default.
    eps = 1e-8
    if args.norm_from_all:
        q_mean = Q.mean(axis=0)
        q_std = Q.std(axis=0) + eps
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + eps
        norm_source = "all_data"
    else:
        q_mean = Q_train.mean(axis=0)
        q_std = Q_train.std(axis=0) + eps
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0) + eps
        norm_source = "train_only"

    train_ds = QXDataset(Q_train, X_train, q_mean, q_std, x_mean, x_std)
    val_ds = QXDataset(Q_val, X_val, q_mean, q_std, x_mean, x_std)
    test_ds = QXDataset(Q_test, X_test, q_mean, q_std, x_mean, x_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SoftArmPosNet(in_dim=in_dim, out_dim=out_dim, width=args.width, depth=args.depth).to(device)
    model.set_normalization(q_mean, q_std, x_mean, x_std)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.eta_min_ratio
    )

    best_val = float("inf")
    best_state = None
    history = []

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = eval_one_epoch(model, val_loader, device, loss_fn)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "q_mean": q_mean,
                "q_std": q_std,
                "x_mean": x_mean,
                "x_std": x_std,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "args": vars(args),
                "normalization_source": norm_source,
            }

        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.4e} | val_loss={val_loss:.4e}")

    train_time_sec = time.time() - t_start
    print(f"Best val loss: {best_val:.4e}")
    print(f"Training time: {train_time_sec:.2f} s")

    # Evaluate best model
    assert best_state is not None
    model.load_state_dict(best_state["model"])
    test_loss = eval_one_epoch(model, test_loader, device, loss_fn)
    val_loss_best = eval_one_epoch(model, val_loader, device, loss_fn)
    print(f"Best-model val loss: {val_loss_best:.4e}")
    print(f"Test loss: {test_loss:.4e}")

    # Physical-space errors on validation and test sets
    X_val_pred = predict_numpy(model, Q_val, device, batch_size=args.eval_batch_size)
    X_test_pred = predict_numpy(model, Q_test, device, batch_size=args.eval_batch_size)
    val_phys = physical_error_metrics(X_val_pred, X_val, prefix="val")
    test_phys = physical_error_metrics(X_test_pred, X_test, prefix="test")

    # Coverage diagnostics in normalised coordinate space
    val_cov = nearest_neighbour_coverage_metrics(Q_train, Q_val, X_val_pred, X_val, q_mean, q_std, prefix="val")
    test_cov = nearest_neighbour_coverage_metrics(Q_train, Q_test, X_test_pred, X_test, q_mean, q_std, prefix="test")

    # Learned-Jacobian diagnostics
    jac_cond = jacobian_condition_metrics(
        model, Q_test, device, num_points=args.jacobian_points, seed=args.seed, prefix="test"
    )
    jac_fd = jacobian_finite_difference_metrics(
        model, Q_test, device, num_points=args.fd_jacobian_points, seed=args.seed,
        eps_scale=args.fd_eps_scale, prefix="test"
    )

    metrics = {
        "dataset_path": args.mat_path,
        "dataset_size": int(N),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "train_size": int(n_train),
        "val_size": int(n_val),
        "test_size": int(n_test),
        "seed": int(args.seed),
        "normalization_source": norm_source,
        "width": int(args.width),
        "depth": int(args.depth),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss_normalised": float(best_val),
        "best_model_val_loss_normalised": float(val_loss_best),
        "test_loss_normalised": float(test_loss),
        "train_time_sec": float(train_time_sec),
        **val_phys,
        **test_phys,
        **val_cov,
        **test_cov,
        **jac_cond,
        **jac_fd,
    }

    print_metric_block("Physical-space test errors", metrics, keys=[
        "test_rmse_euclidean_mm",
        "test_rmse_x_mm",
        "test_rmse_y_mm",
        "test_rmse_z_mm",
        "test_p95_euclidean_mm",
        "test_max_euclidean_mm",
    ])
    print_metric_block("Sparse-coverage metrics", metrics, keys=[
        "test_nn_dist_mean_norm_q",
        "test_nn_dist_p95_norm_q",
        "test_err_nn_dist_corr",
    ])
    print_metric_block("Jacobian metrics", metrics, keys=[
        "test_jacobian_cond_mean",
        "test_jacobian_cond_p95",
        "test_jacobian_fd_rel_fro_mean",
        "test_jacobian_fd_rel_fro_p95",
        "test_jacobian_fd_cosine_mean",
    ])

    # Save checkpoint, metrics, and split information
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "softarm_pos_net.pth")
    torch.save(best_state, save_path)
    print(f"Saved best model to {save_path}")

    metrics_csv_path = os.path.join(args.out_dir, "metrics.csv")
    save_metrics_csv(metrics, metrics_csv_path)
    print(f"Saved metrics to {metrics_csv_path}")

    metrics_json_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON to {metrics_json_path}")

    history_csv_path = os.path.join(args.out_dir, "training_history.csv")
    with open(history_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history to {history_csv_path}")

    split_path = os.path.join(args.out_dir, "split_indices.npz")
    np.savez(split_path, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    print(f"Saved split indices to {split_path}")

    if args.save_predictions:
        pred_path = os.path.join(args.out_dir, "test_predictions.npz")
        np.savez(pred_path, Q_test=Q_test, X_test=X_test, X_pred_test=X_test_pred,
                 err_test=X_test_pred - X_test)
        print(f"Saved test predictions to {pred_path}")

    # Optional evaluation on a fixed external dataset
    if args.eval_mat_path:
        print(f"\nEvaluating external dataset: {args.eval_mat_path}")
        Q_ext, X_ext = load_qx_mat(args.eval_mat_path)
        X_ext_pred = predict_numpy(model, Q_ext, device, batch_size=args.eval_batch_size)
        ext_metrics = physical_error_metrics(X_ext_pred, X_ext, prefix="external")
        ext_cov = nearest_neighbour_coverage_metrics(Q_train, Q_ext, X_ext_pred, X_ext, q_mean, q_std, prefix="external")
        ext_metrics.update(ext_cov)

        ext_csv = os.path.join(args.out_dir, "external_eval_metrics.csv")
        save_metrics_csv(ext_metrics, ext_csv)
        print_metric_block("External-set physical errors", ext_metrics)
        print(f"Saved external evaluation metrics to {ext_csv}")

        if args.save_predictions:
            ext_pred_path = os.path.join(args.out_dir, "external_predictions.npz")
            np.savez(ext_pred_path, Q=Q_ext, X=X_ext, X_pred=X_ext_pred, err=X_ext_pred - X_ext)
            print(f"Saved external predictions to {ext_pred_path}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a soft-arm body-schema MLP and export reproducibility metrics.")
    parser.add_argument("--mat_path", type=str, default="softarm_dataset_mm.mat",
                        help="Path to .mat file containing Q [N,in_dim] and X [N,3].")
    parser.add_argument("--out_dir", type=str, default="checkpoints_softarm")

    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min_ratio", type=float, default=0.1)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--norm_from_all", action="store_true",
                        help="Use full dataset for normalisation, matching the older script. Default: train split only.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--jacobian_points", type=int, default=200,
                        help="Number of test points for Jacobian condition-number statistics. Use 0 to disable.")
    parser.add_argument("--fd_jacobian_points", type=int, default=50,
                        help="Number of test points for autograd-vs-finite-difference Jacobian check. Use 0 to disable.")
    parser.add_argument("--fd_eps_scale", type=float, default=1e-3)

    parser.add_argument("--eval_mat_path", type=str, default="",
                        help="Optional external fixed test .mat file with Q/X for cross-dataset evaluation.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save test predictions and errors as .npz files.")

    args = parser.parse_args()
    main(args)
