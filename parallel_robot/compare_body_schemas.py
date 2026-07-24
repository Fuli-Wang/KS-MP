#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare analytic and learned body schemas on the 6-SPS parallel platform.

The script runs the same KS-MP controller twice while changing only the
body-schema module:

1. analytic geometry with a finite-difference Jacobian;
2. a structure-informed neural forward map with an automatic-differentiation
   Jacobian.

The reference, gains, differential mapping, participation matrix, damping,
integration step, initial pose, target, and evaluation horizon are shared.
Both generated trajectories are also evaluated through the analytic geometry
to separate controller-internal tracking from forward-model discrepancy.

This is an algorithmic body-schema substitution experiment. The learned model
is trained from analytic pose-to-leg-length samples and is not intended to
represent unmodelled physical effects.

Units:
- Translation and leg length: mm
- Orientation: deg
- Jacobian columns: mm/mm for translation and mm/deg for rotation
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Shared controller and platform parameters
# -----------------------------------------------------------------------------
ITERATION_DEFAULT = 1500
DT_DEFAULT = 0.004
SUBMV_T_DEFAULT = 6.0
TRAJ_DEFAULT = "minjerk"
KP_DEFAULT = 100.0
DP_DEFAULT = 0.2
LAM2_DEFAULT = 1e-4
POSE0_DEFAULT = [0.0, 0.0, 1238.87723, 0.0, 0.0, 0.0]
TARGET_DEFAULT = [1249.900, 1255.800, 1329.800, 1351.200, 1330.900, 1303.300]
BQ_DEFAULT = [0.0] * 6
C_DEFAULT = [1.0] * 6
FD_DELTAS_DEFAULT = [1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1]

BASE_RADIUS = 300.0
PLATFORM_RADIUS = 275.0
Z_BASE = 0.0


# -----------------------------------------------------------------------------
# Analytic body schema
# -----------------------------------------------------------------------------
def generate_base_and_platform_points_np(
    base_radius: float = BASE_RADIUS,
    platform_radius: float = PLATFORM_RADIUS,
    z_base: float = Z_BASE,
) -> Tuple[np.ndarray, np.ndarray]:
    theta_base = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    theta_platform = np.deg2rad(np.array([15, 45, 135, 165, 255, 285], dtype=float))

    base_points = np.stack(
        [
            base_radius * np.cos(theta_base),
            base_radius * np.sin(theta_base),
            z_base * np.ones(6),
        ],
        axis=1,
    )

    platform_points = np.stack(
        [
            platform_radius * np.cos(theta_platform),
            platform_radius * np.sin(theta_platform),
            np.zeros(6),
        ],
        axis=1,
    )
    platform_points -= platform_points.mean(axis=0, keepdims=True)
    return base_points, platform_points


BASE_POINTS, PLATFORM_POINTS_LOCAL = generate_base_and_platform_points_np()


def rotation_matrix_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def lengths_from_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,):
        raise ValueError(f"pose must have shape (6,), got {pose.shape}")
    x, y, z, roll, pitch, yaw = pose
    rot = rotation_matrix_zyx(roll, pitch, yaw)
    points_world = (rot @ PLATFORM_POINTS_LOCAL.T).T + np.array([x, y, z])
    return np.linalg.norm(points_world - BASE_POINTS, axis=1)


def jacobian_lengths_pose(
    pose: np.ndarray,
    deltas: np.ndarray = np.asarray(FD_DELTAS_DEFAULT, dtype=float),
) -> np.ndarray:
    """Forward-difference J = dL/dpose, matching pmp_parallel.py."""
    pose = np.asarray(pose, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if pose.shape != (6,) or deltas.shape != (6,):
        raise ValueError("pose and deltas must both have shape (6,)")

    l0 = lengths_from_pose(pose)
    jac = np.zeros((6, 6), dtype=float)
    for j in range(6):
        perturbed = pose.copy()
        perturbed[j] += deltas[j]
        jac[:, j] = (lengths_from_pose(perturbed) - l0) / deltas[j]
    return jac


# -----------------------------------------------------------------------------
# Learned body schema
# -----------------------------------------------------------------------------
def compute_batch_rotation_matrix(euler_angles_rad: torch.Tensor) -> torch.Tensor:
    roll = euler_angles_rad[:, 0]
    pitch = euler_angles_rad[:, 1]
    yaw = euler_angles_rad[:, 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    batch = euler_angles_rad.shape[0]
    rot = torch.zeros(
        (batch, 3, 3),
        dtype=euler_angles_rad.dtype,
        device=euler_angles_rad.device,
    )
    rot[:, 0, 0] = cy * cp
    rot[:, 0, 1] = cy * sp * sr - sy * cr
    rot[:, 0, 2] = cy * sp * cr + sy * sr
    rot[:, 1, 0] = sy * cp
    rot[:, 1, 1] = sy * sp * sr + cy * cr
    rot[:, 1, 2] = sy * sp * cr - cy * sr
    rot[:, 2, 0] = -sp
    rot[:, 2, 1] = cp * sr
    rot[:, 2, 2] = cp * cr
    return rot


class ParallelBodySchemaNet(nn.Module):
    """Structure-informed pose-to-leg-length surrogate."""

    def __init__(self, num_legs: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.num_legs = num_legs
        local_points = torch.tensor(PLATFORM_POINTS_LOCAL, dtype=torch.float32)
        self.register_buffer("platform_points_local", local_points)

        self.fc_output = nn.Sequential(
            nn.Linear(num_legs * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_legs),
            nn.Softplus(),
        )

    def forward(self, pose_candidate: torch.Tensor) -> torch.Tensor:
        translation = pose_candidate[:, :3]
        euler_rad = torch.deg2rad(pose_candidate[:, 3:6])
        rot = compute_batch_rotation_matrix(euler_rad)

        batch = pose_candidate.shape[0]
        points_local = self.platform_points_local.unsqueeze(0).expand(batch, -1, -1)
        points_rot = torch.bmm(rot, points_local.transpose(1, 2)).transpose(1, 2)
        points_global = points_rot + translation.unsqueeze(1)
        return self.fc_output(points_global.reshape(batch, -1))

    def get_jacobian(self, pose_batch: torch.Tensor) -> torch.Tensor:
        """Return the learned Jacobian dL/dpose for one platform pose."""
        if pose_batch.shape != (1, 6):
            raise ValueError(f"pose_batch must have shape (1, 6), got {pose_batch.shape}")
        inputs = pose_batch.detach().clone().requires_grad_(True)
        outputs = self.forward(inputs)
        rows = []
        for i in range(6):
            grad = torch.autograd.grad(
                outputs[0, i],
                inputs,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0]
            rows.append(grad[0])
        return torch.stack(rows, dim=0)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    for key in ("state_dict", "model_state_dict", "model", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    if not checkpoint or not all(isinstance(k, str) for k in checkpoint.keys()):
        raise ValueError("Could not identify a valid state_dict in the checkpoint")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ("module.", "model.", "network."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _infer_hidden_dim(state_dict: Dict[str, torch.Tensor], fallback: int = 256) -> int:
    for key in ("fc_output.0.weight",):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return fallback


def load_model(model_path: str, device: torch.device, hidden_dim: Optional[int]) -> ParallelBodySchemaNet:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        # Fallback for checkpoints saved as full PyTorch modules.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = _extract_state_dict(checkpoint)
    model_hidden = hidden_dim if hidden_dim is not None else _infer_hidden_dim(state_dict)
    model = ParallelBodySchemaNet(hidden_dim=model_hidden).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with the expected body-schema architecture. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Shared controller and evaluation
# -----------------------------------------------------------------------------
def min_jerk_s(t: float, duration: float) -> float:
    tau = np.clip(t / max(duration, 1e-12), 0.0, 1.0)
    return float(10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)


def dls_pinv(jac: np.ndarray, lam2: float) -> np.ndarray:
    """Return J^T (J J^T + lam2 I)^-1."""
    jt = jac.T
    regularised = jac @ jt + lam2 * np.eye(jac.shape[0])
    return jt @ np.linalg.solve(regularised, np.eye(regularised.shape[0]))


def safe_condition_number(jac: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(jac))
        return value if np.isfinite(value) else float("inf")
    except np.linalg.LinAlgError:
        return float("inf")


def relative_frobenius_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference, ord="fro")), 1e-12)
    return float(np.linalg.norm(estimate - reference, ord="fro") / denominator)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class ControllerConfig:
    steps: int
    dt: float
    submv_T: float
    traj: str
    kp_vec: np.ndarray
    dp_vec: np.ndarray
    lam2: float
    use_jt: bool
    bq_diag: np.ndarray
    c_vec: np.ndarray
    fd_deltas: np.ndarray
    timing_warmup: int
    diagnostic_stride: int


@dataclass
class RunResult:
    schema: str
    rows: List[Dict[str, float]]
    summary: Dict[str, Any]
    final_pose: np.ndarray


def make_reference(
    t: float,
    l0_reference: np.ndarray,
    target_lengths: np.ndarray,
    config: ControllerConfig,
) -> np.ndarray:
    if config.traj == "minjerk":
        s = min_jerk_s(t, config.submv_T)
        return l0_reference + s * (target_lengths - l0_reference)
    if config.traj == "legconst":
        return target_lengths.copy()
    raise ValueError(f"Unsupported trajectory for controlled comparison: {config.traj}")


def evaluate_maps_and_jacobians(
    pose: np.ndarray,
    model: ParallelBodySchemaNet,
    device: torch.device,
    fd_deltas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_geom = lengths_from_pose(pose)
    j_geom = jacobian_lengths_pose(pose, fd_deltas)
    pose_t = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        l_nn = model(pose_t).detach().cpu().numpy()[0].astype(float)
    j_nn = model.get_jacobian(pose_t).detach().cpu().numpy().astype(float)
    return l_geom, l_nn, j_geom, j_nn


def run_schema(
    schema: str,
    model: ParallelBodySchemaNet,
    device: torch.device,
    pose0: np.ndarray,
    l0_reference: np.ndarray,
    target_lengths: np.ndarray,
    config: ControllerConfig,
) -> RunResult:
    if schema not in {"analytic", "learned"}:
        raise ValueError("schema must be 'analytic' or 'learned'")

    pose = np.asarray(pose0, dtype=float).copy()
    kp = np.diag(config.kp_vec)
    dp = np.diag(config.dp_vec)
    bq = np.diag(config.bq_diag)
    cmat = np.diag(config.c_vec)

    # Each controller differentiates the forward map used internally.
    if schema == "analytic":
        l_internal_prev = lengths_from_pose(pose)
    else:
        pose_t0 = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            l_internal_prev = model(pose_t0).detach().cpu().numpy()[0].astype(float)
    l_ref_prev = l0_reference.copy()

    rows: List[Dict[str, float]] = []
    core_times_ms: List[float] = []

    internal_errors: List[np.ndarray] = []
    geometry_errors: List[np.ndarray] = []
    model_discrepancies: List[np.ndarray] = []
    cond_internal_values: List[float] = []
    cond_geom_values: List[float] = []
    jac_rel_values: List[float] = []

    for step in range(config.steps):
        t = step * config.dt
        l_ref = make_reference(t, l0_reference, target_lengths, config)
        pose_before = pose.copy()

        # Time only the controller update; common diagnostics are excluded.
        if schema == "learned":
            synchronize_if_needed(device)
        tic = time.perf_counter()

        if schema == "analytic":
            l_internal = lengths_from_pose(pose_before)
            j_internal = jacobian_lengths_pose(pose_before, config.fd_deltas)
        else:
            pose_t = torch.tensor(pose_before, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                l_internal = model(pose_t).detach().cpu().numpy()[0].astype(float)
            j_internal = model.get_jacobian(pose_t).detach().cpu().numpy().astype(float)

        ldot_internal = (l_internal - l_internal_prev) / config.dt
        ldot_ref = (l_ref - l_ref_prev) / config.dt
        e_internal = l_ref - l_internal
        f_task = kp @ e_internal + dp @ (ldot_ref - ldot_internal)

        if config.use_jt:
            raw_pose_rate = j_internal.T @ f_task
        else:
            raw_pose_rate = dls_pinv(j_internal, config.lam2) @ f_task

        pose_rate = (np.eye(6) - bq) @ (cmat @ raw_pose_rate)
        pose = pose_before + pose_rate * config.dt

        if schema == "learned":
            synchronize_if_needed(device)
        core_ms = (time.perf_counter() - tic) * 1000.0
        core_times_ms.append(core_ms)

        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(pose_rate)):
            raise FloatingPointError(
                f"Non-finite state in {schema} controller at step {step}. "
                "Check target, model coverage, gains, and Jacobian conditioning."
            )

        # Evaluate both forward maps at the same pre-update state. Jacobian
        # cross-checks are evaluated at the requested diagnostic stride.
        diagnostic_step = (step % config.diagnostic_stride == 0) or (step == config.steps - 1)
        if schema == "analytic":
            l_geom = l_internal
            j_geom = j_internal
            pose_t_diag = torch.tensor(
                pose_before, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                l_nn = model(pose_t_diag).detach().cpu().numpy()[0].astype(float)
            j_nn = (
                model.get_jacobian(pose_t_diag).detach().cpu().numpy().astype(float)
                if diagnostic_step
                else None
            )
        else:
            l_nn = l_internal
            j_nn = j_internal
            l_geom = lengths_from_pose(pose_before)
            j_geom = (
                jacobian_lengths_pose(pose_before, config.fd_deltas)
                if diagnostic_step
                else None
            )

        e_geom = l_ref - l_geom
        model_error = l_nn - l_geom
        cond_internal = safe_condition_number(j_internal)
        if diagnostic_step and j_geom is not None and j_nn is not None:
            cond_geom = safe_condition_number(j_geom)
            jac_rel = relative_frobenius_error(j_nn, j_geom)
            cond_geom_values.append(cond_geom)
            jac_rel_values.append(jac_rel)
        else:
            cond_geom = float("nan")
            jac_rel = float("nan")

        internal_errors.append(e_internal)
        geometry_errors.append(e_geom)
        model_discrepancies.append(model_error)
        cond_internal_values.append(cond_internal)

        row: Dict[str, float] = {"step": float(step), "time_s": float(t)}
        for i in range(6):
            row[f"Lref{i+1}_mm"] = float(l_ref[i])
            row[f"Linternal{i+1}_mm"] = float(l_internal[i])
            row[f"Lgeom{i+1}_mm"] = float(l_geom[i])
            row[f"Lnn{i+1}_mm"] = float(l_nn[i])
            row[f"e_internal{i+1}_mm"] = float(e_internal[i])
            row[f"e_geom{i+1}_mm"] = float(e_geom[i])
            row[f"nn_minus_geom{i+1}_mm"] = float(model_error[i])
            row[f"pose_rate{i+1}"] = float(pose_rate[i])
        for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), pose_before):
            row[f"pose_{name}"] = float(value)
        row["internal_error_l2_mm"] = float(np.linalg.norm(e_internal))
        row["geometry_error_l2_mm"] = float(np.linalg.norm(e_geom))
        row["model_discrepancy_l2_mm"] = float(np.linalg.norm(model_error))
        row["cond_internal"] = cond_internal
        row["cond_geometry"] = cond_geom
        row["jacobian_relative_fro_error"] = jac_rel
        row["core_step_ms"] = float(core_ms)
        rows.append(row)

        l_internal_prev = l_internal.copy()
        l_ref_prev = l_ref.copy()

    internal_error_arr = np.asarray(internal_errors, dtype=float)
    geometry_error_arr = np.asarray(geometry_errors, dtype=float)
    model_error_arr = np.asarray(model_discrepancies, dtype=float)
    cond_internal_arr = np.asarray(cond_internal_values, dtype=float)
    cond_geom_arr = np.asarray(cond_geom_values, dtype=float)
    jac_rel_arr = np.asarray(jac_rel_values, dtype=float)
    core_arr = np.asarray(core_times_ms, dtype=float)

    # Evaluate the final state after all Euler updates.
    l_geom_final, l_nn_final, j_geom_final, j_nn_final = evaluate_maps_and_jacobians(
        pose, model, device, config.fd_deltas
    )
    l_internal_final = l_geom_final if schema == "analytic" else l_nn_final
    e_internal_final_target = target_lengths - l_internal_final
    e_geom_final_target = target_lengths - l_geom_final

    warmup = min(max(config.timing_warmup, 0), max(len(core_arr) - 1, 0))
    core_eval = core_arr[warmup:] if len(core_arr) else core_arr

    def rms_euclidean(errors: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))

    def component_rmse(errors: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors**2)))

    def finite_percentile(values: np.ndarray, q: float) -> float:
        finite = values[np.isfinite(values)]
        return float(np.percentile(finite, q)) if finite.size else float("inf")

    summary: Dict[str, Any] = {
        "schema": schema,
        "controller_forward_map": "geometry" if schema == "analytic" else "learned",
        "controller_jacobian": "finite_difference_geometry" if schema == "analytic" else "autodiff_learned",
        "steps": int(config.steps),
        "dt_s": float(config.dt),
        "duration_s": float(config.steps * config.dt),
        "mapping": "JT" if config.use_jt else "DLS",
        "lam2": float(config.lam2),
        "internal_tracking_rms_euclidean_mm": rms_euclidean(internal_error_arr),
        "internal_tracking_component_rmse_mm": component_rmse(internal_error_arr),
        "geometry_replay_tracking_rms_euclidean_mm": rms_euclidean(geometry_error_arr),
        "geometry_replay_tracking_component_rmse_mm": component_rmse(geometry_error_arr),
        "internal_final_target_l2_mm": float(np.linalg.norm(e_internal_final_target)),
        "internal_final_target_component_rmse_mm": float(np.sqrt(np.mean(e_internal_final_target**2))),
        "geometry_final_target_l2_mm": float(np.linalg.norm(e_geom_final_target)),
        "geometry_final_target_component_rmse_mm": float(np.sqrt(np.mean(e_geom_final_target**2))),
        "forward_model_discrepancy_rms_euclidean_mm": rms_euclidean(model_error_arr),
        "forward_model_discrepancy_component_rmse_mm": component_rmse(model_error_arr),
        "forward_model_discrepancy_final_l2_mm": float(np.linalg.norm(l_nn_final - l_geom_final)),
        "jacobian_relative_fro_mean": float(np.mean(jac_rel_arr)),
        "jacobian_relative_fro_p95": float(np.percentile(jac_rel_arr, 95)),
        "jacobian_relative_fro_max": float(np.max(jac_rel_arr)),
        "cond_internal_mean": float(np.mean(cond_internal_arr)),
        "cond_internal_p95": finite_percentile(cond_internal_arr, 95),
        "cond_internal_max": float(np.max(cond_internal_arr)),
        "cond_geometry_mean": float(np.mean(cond_geom_arr)),
        "cond_geometry_p95": finite_percentile(cond_geom_arr, 95),
        "cond_geometry_max": float(np.max(cond_geom_arr)),
        "runtime_core_mean_ms": float(np.mean(core_eval)),
        "runtime_core_median_ms": float(np.median(core_eval)),
        "runtime_core_p95_ms": float(np.percentile(core_eval, 95)),
        "runtime_core_p99_ms": float(np.percentile(core_eval, 99)),
        "runtime_core_max_ms": float(np.max(core_eval)),
        "runtime_mean_frequency_hz": float(1000.0 / max(np.mean(core_eval), 1e-12)),
        "runtime_warmup_steps": int(warmup),
        "final_pose_x_mm": float(pose[0]),
        "final_pose_y_mm": float(pose[1]),
        "final_pose_z_mm": float(pose[2]),
        "final_pose_roll_deg": float(pose[3]),
        "final_pose_pitch_deg": float(pose[4]),
        "final_pose_yaw_deg": float(pose[5]),
        "final_geom_cond": safe_condition_number(j_geom_final),
        "final_learned_cond": safe_condition_number(j_nn_final),
        "final_jacobian_relative_fro_error": relative_frobenius_error(j_nn_final, j_geom_final),
    }

    return RunResult(schema=schema, rows=rows, summary=summary, final_pose=pose)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_dict_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("inf")


def build_pairwise_summary(analytic: RunResult, learned: RunResult) -> Dict[str, Any]:
    a = analytic.summary
    l = learned.summary
    return {
        "geometry_replay_rms_analytic_mm": a["geometry_replay_tracking_rms_euclidean_mm"],
        "geometry_replay_rms_learned_mm": l["geometry_replay_tracking_rms_euclidean_mm"],
        "geometry_replay_rms_learned_minus_analytic_mm": (
            l["geometry_replay_tracking_rms_euclidean_mm"]
            - a["geometry_replay_tracking_rms_euclidean_mm"]
        ),
        "geometry_replay_rms_ratio_learned_over_analytic": safe_ratio(
            l["geometry_replay_tracking_rms_euclidean_mm"],
            a["geometry_replay_tracking_rms_euclidean_mm"],
        ),
        "geometry_final_l2_analytic_mm": a["geometry_final_target_l2_mm"],
        "geometry_final_l2_learned_mm": l["geometry_final_target_l2_mm"],
        "geometry_final_l2_learned_minus_analytic_mm": (
            l["geometry_final_target_l2_mm"] - a["geometry_final_target_l2_mm"]
        ),
        "runtime_mean_analytic_ms": a["runtime_core_mean_ms"],
        "runtime_mean_learned_ms": l["runtime_core_mean_ms"],
        "runtime_ratio_learned_over_analytic": safe_ratio(
            l["runtime_core_mean_ms"], a["runtime_core_mean_ms"]
        ),
        "learned_internal_rms_mm": l["internal_tracking_rms_euclidean_mm"],
        "learned_geometry_replay_rms_mm": l["geometry_replay_tracking_rms_euclidean_mm"],
        "learned_internal_vs_geometry_rms_gap_mm": (
            l["geometry_replay_tracking_rms_euclidean_mm"]
            - l["internal_tracking_rms_euclidean_mm"]
        ),
        "learned_forward_discrepancy_rms_mm": l[
            "forward_model_discrepancy_rms_euclidean_mm"
        ],
        "learned_jacobian_relative_fro_mean": l["jacobian_relative_fro_mean"],
        "learned_jacobian_relative_fro_p95": l["jacobian_relative_fro_p95"],
    }


def print_run_summary(summary: Dict[str, Any]) -> None:
    print(f"\n[{summary['schema'].upper()}]")
    print(
        "  internal tracking RMS (Euclidean): "
        f"{summary['internal_tracking_rms_euclidean_mm']:.6f} mm"
    )
    print(
        "  geometry-replay tracking RMS:      "
        f"{summary['geometry_replay_tracking_rms_euclidean_mm']:.6f} mm"
    )
    print(
        "  final geometry target L2 error:    "
        f"{summary['geometry_final_target_l2_mm']:.6f} mm"
    )
    print(
        "  NN-geometry discrepancy RMS:       "
        f"{summary['forward_model_discrepancy_rms_euclidean_mm']:.6f} mm"
    )
    print(
        "  Jacobian relative Frobenius mean:  "
        f"{summary['jacobian_relative_fro_mean']:.6e}"
    )
    print(
        "  internal Jacobian cond mean/p95:   "
        f"{summary['cond_internal_mean']:.3f} / {summary['cond_internal_p95']:.3f}"
    )
    print(
        "  core runtime mean/p95:              "
        f"{summary['runtime_core_mean_ms']:.4f} / {summary['runtime_core_p95_ms']:.4f} ms"
    )


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def expand_six(values: Optional[List[float]], scalar: float, name: str) -> np.ndarray:
    if values is None:
        return np.full(6, float(scalar), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(6, float(arr[0]), dtype=float)
    if arr.size != 6:
        raise ValueError(f"{name} must contain 1 or 6 values")
    return arr


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare analytic and learned body schemas on the 6-SPS platform"
    )
    parser.add_argument(
        "--model",
        default="models/parallel_body_schema.pth",
        help="learned body-schema checkpoint (.pth/.pt)",
    )
    parser.add_argument("--hidden-dim", type=int, default=None, help="override inferred hidden width")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")

    parser.add_argument("--pose0", type=float, nargs=6, default=POSE0_DEFAULT)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target", type=float, nargs=6, default=None, help="target leg lengths [mm]")
    target_group.add_argument(
        "--target-pose",
        type=float,
        nargs=6,
        default=None,
        help="target platform pose; target lengths are generated by the analytic geometry",
    )

    parser.add_argument("--steps", type=int, default=ITERATION_DEFAULT)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--submv-T", dest="submv_T", type=float, default=SUBMV_T_DEFAULT)
    parser.add_argument("--traj", choices=["minjerk", "legconst"], default=TRAJ_DEFAULT)

    parser.add_argument("--kp", type=float, default=KP_DEFAULT)
    parser.add_argument("--kp-vec", type=float, nargs="+", default=None)
    parser.add_argument("--dp", type=float, default=DP_DEFAULT)
    parser.add_argument("--dp-vec", type=float, nargs="+", default=None)
    parser.add_argument("--use-jt", action="store_true", default=False)
    parser.add_argument("--lam2", type=float, default=LAM2_DEFAULT)
    parser.add_argument("--bq-diag", type=float, nargs="+", default=BQ_DEFAULT)
    parser.add_argument("--c-vec", type=float, nargs="+", default=C_DEFAULT)
    parser.add_argument("--fd-deltas", type=float, nargs=6, default=FD_DELTAS_DEFAULT)

    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument(
        "--diagnostic-stride",
        type=int,
        default=10,
        help="evaluate learned-vs-geometry Jacobian diagnostics every N steps",
    )
    parser.add_argument("--out-dir", default="body_schema_comparison_results")
    parser.add_argument("--run-name", default="comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.submv_T <= 0.0:
        raise ValueError("--submv-T must be positive")
    if args.lam2 < 0.0:
        raise ValueError("--lam2 must be non-negative")
    if args.diagnostic_stride <= 0:
        raise ValueError("--diagnostic-stride must be positive")

    device = resolve_device(args.device)
    model = load_model(args.model, device=device, hidden_dim=args.hidden_dim)

    pose0 = np.asarray(args.pose0, dtype=float)
    l0_reference = lengths_from_pose(pose0)
    if args.target_pose is not None:
        target_pose = np.asarray(args.target_pose, dtype=float)
        target_lengths = lengths_from_pose(target_pose)
    else:
        target_pose = None
        target_lengths = np.asarray(
            args.target if args.target is not None else TARGET_DEFAULT, dtype=float
        )

    kp_vec = expand_six(args.kp_vec, args.kp, "kp")
    dp_vec = expand_six(args.dp_vec, args.dp, "dp")
    bq_diag = expand_six(args.bq_diag, 0.0, "bq_diag")
    c_vec = expand_six(args.c_vec, 1.0, "c_vec")

    config = ControllerConfig(
        steps=int(args.steps),
        dt=float(args.dt),
        submv_T=float(args.submv_T),
        traj=args.traj,
        kp_vec=kp_vec,
        dp_vec=dp_vec,
        lam2=float(args.lam2),
        use_jt=bool(args.use_jt),
        bq_diag=bq_diag,
        c_vec=c_vec,
        fd_deltas=np.asarray(args.fd_deltas, dtype=float),
        timing_warmup=int(args.timing_warmup),
        diagnostic_stride=max(1, int(args.diagnostic_stride)),
    )

    # Report the initial learned-versus-analytic forward-map discrepancy.
    pose0_t = torch.tensor(pose0, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        l0_nn = model(pose0_t).detach().cpu().numpy()[0].astype(float)
    initial_discrepancy = l0_nn - l0_reference

    print("[info] Analytic-versus-learned 6-SPS body-schema comparison")
    print(f"[info] Device: {device}")
    print(f"[info] Model: {args.model}")
    print(f"[info] Initial pose [mm,deg]: {pose0.tolist()}")
    print(f"[info] Geometry initial lengths [mm]: {np.round(l0_reference, 6).tolist()}")
    print(f"[info] Learned initial lengths [mm]:  {np.round(l0_nn, 6).tolist()}")
    print(
        "[info] Initial learned-geometry discrepancy L2: "
        f"{np.linalg.norm(initial_discrepancy):.6f} mm"
    )
    print(f"[info] Target lengths [mm]: {target_lengths.tolist()}")
    if target_pose is not None:
        print(f"[info] Target pose [mm,deg]: {target_pose.tolist()}")
    print(
        f"[info] Shared controller: traj={config.traj}, steps={config.steps}, "
        f"dt={config.dt}, T={config.submv_T}, mapping={'JT' if config.use_jt else 'DLS'}, "
        f"lam2={config.lam2}"
    )
    print(
        f"[info] Shared gains: Kp={config.kp_vec.tolist()}, Dp={config.dp_vec.tolist()}, "
        f"BQ={config.bq_diag.tolist()}, C={config.c_vec.tolist()}"
    )

    analytic = run_schema(
        "analytic", model, device, pose0, l0_reference, target_lengths, config
    )
    learned = run_schema(
        "learned", model, device, pose0, l0_reference, target_lengths, config
    )

    # Attach common initial diagnostics to both summaries.
    for result in (analytic, learned):
        result.summary["initial_forward_model_discrepancy_l2_mm"] = float(
            np.linalg.norm(initial_discrepancy)
        )
        result.summary["initial_forward_model_discrepancy_component_rmse_mm"] = float(
            np.sqrt(np.mean(initial_discrepancy**2))
        )

    pairwise = build_pairwise_summary(analytic, learned)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_name
    summary_path = out_dir / f"{stem}_summary.csv"
    pairwise_path = out_dir / f"{stem}_pairwise.csv"
    analytic_path = out_dir / f"{stem}_analytic_trajectory.csv"
    learned_path = out_dir / f"{stem}_learned_trajectory.csv"
    metadata_path = out_dir / f"{stem}_metadata.json"

    write_dict_rows(summary_path, [analytic.summary, learned.summary])
    write_dict_rows(pairwise_path, [pairwise])
    write_dict_rows(analytic_path, analytic.rows)
    write_dict_rows(learned_path, learned.rows)

    metadata = {
        "model_path": str(Path(args.model).resolve()),
        "device": str(device),
        "pose0": pose0.tolist(),
        "geometry_initial_lengths_mm": l0_reference.tolist(),
        "learned_initial_lengths_mm": l0_nn.tolist(),
        "initial_nn_minus_geometry_mm": initial_discrepancy.tolist(),
        "target_lengths_mm": target_lengths.tolist(),
        "target_pose": target_pose.tolist() if target_pose is not None else None,
        "controller": {
            "steps": config.steps,
            "dt_s": config.dt,
            "submovement_duration_s": config.submv_T,
            "trajectory": config.traj,
            "mapping": "JT" if config.use_jt else "DLS",
            "lambda_squared": config.lam2,
            "kp_vec": config.kp_vec.tolist(),
            "dp_vec": config.dp_vec.tolist(),
            "bq_diag": config.bq_diag.tolist(),
            "c_vec": config.c_vec.tolist(),
            "finite_difference_deltas": config.fd_deltas.tolist(),
        },
        "evaluation_note": (
            "Both trajectories are evaluated through the same analytic geometry. "
            "The learned checkpoint was trained from analytic pose-leg-length pairs, "
            "so this is an algorithmic body-schema substitution test, not a test of "
            "unmodelled physical-effect compensation."
        ),
        "output_files": {
            "summary": str(summary_path),
            "pairwise": str(pairwise_path),
            "analytic_trajectory": str(analytic_path),
            "learned_trajectory": str(learned_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=True)

    print_run_summary(analytic.summary)
    print_run_summary(learned.summary)
    print("\n[PAIRWISE]")
    print(
        "  learned/analytic geometry-replay RMS ratio: "
        f"{pairwise['geometry_replay_rms_ratio_learned_over_analytic']:.6f}"
    )
    print(
        "  learned/analytic runtime ratio:             "
        f"{pairwise['runtime_ratio_learned_over_analytic']:.6f}"
    )
    print(
        "  learned internal-to-geometry RMS gap:       "
        f"{pairwise['learned_internal_vs_geometry_rms_gap_mm']:.6f} mm"
    )

    print("\n[info] Files saved:")
    for path in (summary_path, pairwise_path, analytic_path, learned_path, metadata_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
