#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeated benchmark for the parallel-robot KS-MP implementation.

The script compares three methods under the same geometric 6-SPS body schema
and minimum-jerk leg-length reference:

1. KS-MP with damped least-squares mapping;
2. leg-length PID-style tracking with DLS pose-rate mapping;
3. iterative DLS pose solving at each reference point.

Run one method per invocation so that method order can be interleaved across
repeated trials. Reference generation and diagnostic-only calculations are
excluded from the timed controller section where applicable.

The reported deviations and timings are algorithmic, model-based quantities.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Import the parallel-robot implementation from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
PARALLEL_ROBOT_DIR = REPO_ROOT / "parallel_robot"
if str(PARALLEL_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(PARALLEL_ROBOT_DIR))

import pmp_parallel as pp


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------
def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(value: float) -> float:
    return float(value) / 1e6


def ns_to_us(value: float) -> float:
    return float(value) / 1e3


def dls_pinv(J: np.ndarray, lam2: float) -> np.ndarray:
    """Return J^T (J J^T + lambda^2 I)^-1."""
    JT = J.T
    regularised = J @ JT + float(lam2) * np.eye(J.shape[0])
    return JT @ np.linalg.solve(
        regularised,
        np.eye(regularised.shape[0]),
    )


def make_diag(values: Iterable[float], n: int) -> np.ndarray:
    v = np.asarray(list(values), dtype=float)
    if v.size == 1:
        v = np.full(n, float(v[0]), dtype=float)
    if v.size != n:
        raise ValueError(f"Expected 1 or {n} values, got {v.size}")
    return np.diag(v)


def safe_cond_from_svd(s: np.ndarray, eps: float = 1e-12) -> float:
    if s.size == 0:
        return float("nan")
    smax = float(np.max(s))
    smin = float(np.min(s))
    if smin <= eps:
        return float("inf")
    return smax / smin


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def reference_minjerk(y0: np.ndarray, y_target: np.ndarray, t: float, T: float) -> np.ndarray:
    s = pp.min_jerk_s(t, T)
    return y0 + s * (y_target - y0)


def sanitise_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "parallel_run"


def serialise_vector(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def resolve_c_vector(cli: argparse.Namespace) -> np.ndarray:
    if cli.c_vec is None:
        return np.ones(6, dtype=float)
    v = np.asarray(cli.c_vec, dtype=float)
    if v.size == 1:
        v = np.full(6, float(v[0]), dtype=float)
    if v.size != 6:
        raise ValueError("--c-vec must contain either 1 or 6 values")
    return v


def get_ksmp_matrices(cli: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Kp = make_diag(cli.kp_vec if cli.kp_vec is not None else [cli.kp], 6)
    Dp = make_diag(cli.dp_vec if cli.dp_vec is not None else [cli.dp], 6)
    C = np.diag(resolve_c_vector(cli))
    BQ = make_diag(cli.bq_diag, 6)
    return Kp, Dp, C, BQ


def make_tracking_gain(cli: argparse.Namespace, scalar_name: str, vector_name: str) -> np.ndarray:
    vector = getattr(cli, vector_name)
    if vector is not None:
        return make_diag(vector, 6)
    return np.eye(6) * float(getattr(cli, scalar_name))


def compute_local_stability_metrics(
    J: np.ndarray,
    Kp: np.ndarray,
    Dp: np.ndarray,
    C: np.ndarray,
    BQ: np.ndarray,
    dt: float,
    lam2: float,
) -> Dict[str, float]:
    """Controller-internal local discrete-time diagnostic; excluded from timing."""
    M = dls_pinv(J, lam2=lam2)
    H = J @ (np.eye(J.shape[1]) - BQ) @ C @ M
    I = np.eye(J.shape[0])
    A_d = np.block([
        [I - dt * (H @ Kp) - (H @ Dp), H @ Dp],
        [I, np.zeros_like(I)],
    ])
    H_eigs = np.linalg.eigvals(H)
    H_abs = np.abs(H_eigs)
    return {
        "rho_Ad": spectral_radius(A_d),
        "H_eig_abs_min": float(np.min(H_abs)),
        "H_eig_abs_max": float(np.max(H_abs)),
        "H_cond_svd": safe_cond_from_svd(np.linalg.svd(H, compute_uv=False)),
    }


def timing_statistics(step_ms: np.ndarray, warmup_steps: int) -> Dict[str, float]:
    step_ms = np.asarray(step_ms, dtype=float)
    if step_ms.size == 0:
        raise ValueError("No timing samples were recorded")

    warmup = int(max(0, warmup_steps))
    if warmup >= step_ms.size:
        raise ValueError(
            f"Warm-up ({warmup}) must be smaller than the number of steps ({step_ms.size})"
        )

    evaluated = step_ms[warmup:]
    return {
        "runtime_measured_steps": int(step_ms.size),
        "runtime_warmup_steps": int(warmup),
        "runtime_evaluated_steps": int(evaluated.size),
        "mean_step_ms": float(np.mean(evaluated)),
        "median_step_ms": float(np.median(evaluated)),
        "p95_step_ms": float(np.percentile(evaluated, 95)),
        "p99_step_ms": float(np.percentile(evaluated, 99)),
        "max_step_ms": float(np.max(evaluated)),
    }


# -----------------------------------------------------------------------------
# KS-MP DLS benchmark
# -----------------------------------------------------------------------------
def run_ksmp_dls(
    cli: argparse.Namespace,
    target_lengths: np.ndarray,
    log_path: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    pose = np.asarray(cli.pose0, dtype=float).copy()
    L0 = pp.lengths_from_pose(pose)
    target_lengths = np.asarray(target_lengths, dtype=float)

    L_prev = L0.copy()
    L_ref_prev = L0.copy()
    Kp, Dp, C, BQ = get_ksmp_matrices(cli)

    fieldnames = [
        "time_s",
        *[f"L{i}" for i in range(1, 7)],
        *[f"Lref{i}" for i in range(1, 7)],
        *[f"err{i}" for i in range(1, 7)],
        "err_norm_mm",
        *[f"F{i}" for i in range(1, 7)],
        *[f"posedot{i}" for i in range(1, 7)],
        "x", "y", "z", "roll", "pitch", "yaw",
        "sigma_min_J", "sigma_max_J", "cond_J",
        "rho_Ad", "H_eig_abs_min", "H_eig_abs_max", "H_cond_svd",
        "step_us", "geom_us", "jac_us", "map_update_us", "diag_us",
    ]

    step_ns_values: List[int] = []
    err_norms: List[float] = []
    cond_values: List[float] = []
    rho_values: List[float] = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(int(cli.steps)):
            t = k * float(cli.dt)
            # Reference generation is intentionally outside the timed section.
            L_ref = reference_minjerk(L0, target_lengths, t, float(cli.submv_T))

            t0 = now_ns()

            tg0 = now_ns()
            L_cur = pp.lengths_from_pose(pose)
            tg1 = now_ns()

            dt = float(cli.dt)
            Ldot_cur = (L_cur - L_prev) / dt
            Ldot_ref = (L_ref - L_ref_prev) / dt
            e = L_ref - L_cur
            F_task = Kp @ e + Dp @ (Ldot_ref - Ldot_cur)

            tj0 = now_ns()
            J = pp.jacobian_lengths_pose(pose).astype(float)
            tj1 = now_ns()

            tm0 = now_ns()
            Jpinv = dls_pinv(J, lam2=float(cli.lam2))
            tau_pose = Jpinv @ F_task
            posedot = (np.eye(6) - BQ) @ (C @ tau_pose)
            pose_next = pose + posedot * dt
            tm1 = now_ns()

            # FAIR TIMING BOUNDARY: diagnostics below are excluded.
            t1 = now_ns()

            td0 = now_ns()
            svals = np.linalg.svd(J, compute_uv=False)
            condJ = safe_cond_from_svd(svals)
            diag = compute_local_stability_metrics(
                J=J,
                Kp=Kp,
                Dp=Dp,
                C=C,
                BQ=BQ,
                dt=dt,
                lam2=float(cli.lam2),
            )
            td1 = now_ns()

            err_norm = float(np.linalg.norm(e))
            err_norms.append(err_norm)
            step_ns_values.append(t1 - t0)
            cond_values.append(condJ)
            rho_values.append(diag["rho_Ad"])

            row: Dict[str, float] = {"time_s": t}
            row.update({f"L{i+1}": L_cur[i] for i in range(6)})
            row.update({f"Lref{i+1}": L_ref[i] for i in range(6)})
            row.update({f"err{i+1}": e[i] for i in range(6)})
            row["err_norm_mm"] = err_norm
            row.update({f"F{i+1}": F_task[i] for i in range(6)})
            row.update({f"posedot{i+1}": posedot[i] for i in range(6)})
            row.update({
                "x": pose_next[0], "y": pose_next[1], "z": pose_next[2],
                "roll": pose_next[3], "pitch": pose_next[4], "yaw": pose_next[5],
                "sigma_min_J": float(np.min(svals)),
                "sigma_max_J": float(np.max(svals)),
                "cond_J": condJ,
                "rho_Ad": diag["rho_Ad"],
                "H_eig_abs_min": diag["H_eig_abs_min"],
                "H_eig_abs_max": diag["H_eig_abs_max"],
                "H_cond_svd": diag["H_cond_svd"],
                "step_us": ns_to_us(t1 - t0),
                "geom_us": ns_to_us(tg1 - tg0),
                "jac_us": ns_to_us(tj1 - tj0),
                "map_update_us": ns_to_us(tm1 - tm0),
                "diag_us": ns_to_us(td1 - td0),
            })
            writer.writerow(row)

            pose = pose_next
            L_prev = L_cur.copy()
            L_ref_prev = L_ref.copy()

    errors = np.asarray(err_norms, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    metrics: Dict[str, float] = {
        "method": "KS-MP-DLS",
        "rms_error_mm": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_mm": float(np.mean(errors)),
        "max_error_mm": float(np.max(errors)),
        "final_error_mm": float(errors[-1]),
        "max_cond_J": float(np.nanmax(cond_values)),
        "max_rho_Ad": float(np.max(rho_values)),
        "mean_iterations": float("nan"),
        "failure_count": 0,
    }
    return metrics, step_ms


# -----------------------------------------------------------------------------
# Leg-length tracking baseline
# -----------------------------------------------------------------------------
def run_leg_pid(
    cli: argparse.Namespace,
    target_lengths: np.ndarray,
    log_path: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    pose = np.asarray(cli.pose0, dtype=float).copy()
    L0 = pp.lengths_from_pose(pose)
    target_lengths = np.asarray(target_lengths, dtype=float)

    K_L = make_tracking_gain(cli, "leg_kp", "leg_kp_vec")
    D_L = make_tracking_gain(cli, "leg_dp", "leg_dp_vec")
    I_L = make_tracking_gain(cli, "leg_ki", "leg_ki_vec")

    L_prev = L0.copy()
    L_ref_prev = L0.copy()
    e_int = np.zeros(6, dtype=float)

    fieldnames = [
        "time_s",
        *[f"L{i}" for i in range(1, 7)],
        *[f"Lref{i}" for i in range(1, 7)],
        *[f"err{i}" for i in range(1, 7)],
        "err_norm_mm",
        *[f"vLcmd{i}" for i in range(1, 7)],
        *[f"posedot{i}" for i in range(1, 7)],
        "x", "y", "z", "roll", "pitch", "yaw",
        "sigma_min_J", "sigma_max_J", "cond_J",
        "step_us", "geom_us", "jac_us", "map_update_us",
    ]

    step_ns_values: List[int] = []
    err_norms: List[float] = []
    cond_values: List[float] = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(int(cli.steps)):
            t = k * float(cli.dt)
            L_ref = reference_minjerk(L0, target_lengths, t, float(cli.submv_T))

            t0 = now_ns()

            tg0 = now_ns()
            L_cur = pp.lengths_from_pose(pose)
            tg1 = now_ns()

            dt = float(cli.dt)
            Ldot_cur = (L_cur - L_prev) / dt
            Ldot_ref = (L_ref - L_ref_prev) / dt
            e = L_ref - L_cur

            e_int = e_int + e * dt
            if float(cli.leg_integral_clip) > 0.0:
                clip = float(cli.leg_integral_clip)
                e_int = np.clip(e_int, -clip, clip)

            vL_cmd = Ldot_ref + K_L @ e + D_L @ (Ldot_ref - Ldot_cur) + I_L @ e_int

            tj0 = now_ns()
            J = pp.jacobian_lengths_pose(pose).astype(float)
            tj1 = now_ns()

            tm0 = now_ns()
            posedot = dls_pinv(J, lam2=float(cli.leg_lam2)) @ vL_cmd
            pose_next = pose + posedot * dt
            tm1 = now_ns()

            t1 = now_ns()

            # Excluded from timing.
            svals = np.linalg.svd(J, compute_uv=False)
            condJ = safe_cond_from_svd(svals)

            err_norm = float(np.linalg.norm(e))
            err_norms.append(err_norm)
            step_ns_values.append(t1 - t0)
            cond_values.append(condJ)

            row: Dict[str, float] = {"time_s": t}
            row.update({f"L{i+1}": L_cur[i] for i in range(6)})
            row.update({f"Lref{i+1}": L_ref[i] for i in range(6)})
            row.update({f"err{i+1}": e[i] for i in range(6)})
            row["err_norm_mm"] = err_norm
            row.update({f"vLcmd{i+1}": vL_cmd[i] for i in range(6)})
            row.update({f"posedot{i+1}": posedot[i] for i in range(6)})
            row.update({
                "x": pose_next[0], "y": pose_next[1], "z": pose_next[2],
                "roll": pose_next[3], "pitch": pose_next[4], "yaw": pose_next[5],
                "sigma_min_J": float(np.min(svals)),
                "sigma_max_J": float(np.max(svals)),
                "cond_J": condJ,
                "step_us": ns_to_us(t1 - t0),
                "geom_us": ns_to_us(tg1 - tg0),
                "jac_us": ns_to_us(tj1 - tj0),
                "map_update_us": ns_to_us(tm1 - tm0),
            })
            writer.writerow(row)

            pose = pose_next
            L_prev = L_cur.copy()
            L_ref_prev = L_ref.copy()

    errors = np.asarray(err_norms, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    metrics: Dict[str, float] = {
        "method": "Leg-length-PID-DLS-IK-baseline",
        "rms_error_mm": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_mm": float(np.mean(errors)),
        "max_error_mm": float(np.max(errors)),
        "final_error_mm": float(errors[-1]),
        "max_cond_J": float(np.nanmax(cond_values)),
        "max_rho_Ad": float("nan"),
        "mean_iterations": float("nan"),
        "failure_count": 0,
    }
    return metrics, step_ms


# -----------------------------------------------------------------------------
# Iterative DLS pose-solve baseline
# -----------------------------------------------------------------------------
def solve_pose_from_lengths_dls_single(
    pose_seed: np.ndarray,
    L_target: np.ndarray,
    lam2: float,
    tol_mm: float,
    max_iter: int,
    step_scale: float,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    pose = np.asarray(pose_seed, dtype=float).copy()
    L_target = np.asarray(L_target, dtype=float)

    success = False
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        L_cur = pp.lengths_from_pose(pose)
        e = L_target - L_cur
        if float(np.linalg.norm(e)) <= float(tol_mm):
            success = True
            break
        J = pp.jacobian_lengths_pose(pose).astype(float)
        dpose = dls_pinv(J, lam2=float(lam2)) @ e
        pose = pose + float(step_scale) * dpose

    L_cur = pp.lengths_from_pose(pose)
    err_norm = float(np.linalg.norm(L_target - L_cur))
    success = success or (err_norm <= float(tol_mm))
    return pose, L_cur, err_norm, iteration, success


def run_pose_solve(
    cli: argparse.Namespace,
    target_lengths: np.ndarray,
    log_path: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    pose = np.asarray(cli.pose0, dtype=float).copy()
    L0 = pp.lengths_from_pose(pose)
    target_lengths = np.asarray(target_lengths, dtype=float)

    fieldnames = [
        "time_s",
        *[f"Lref{i}" for i in range(1, 7)],
        *[f"Lsol{i}" for i in range(1, 7)],
        "xsol", "ysol", "zsol", "rollsol", "pitchsol", "yawsol",
        "err_norm_mm", "iterations", "success", "step_us",
    ]

    step_ns_values: List[int] = []
    err_norms: List[float] = []
    iteration_values: List[float] = []
    failures = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(int(cli.steps)):
            t = k * float(cli.dt)
            L_ref = reference_minjerk(L0, target_lengths, t, float(cli.submv_T))

            t0 = now_ns()
            pose, L_sol, err_norm, iterations, success = solve_pose_from_lengths_dls_single(
                pose_seed=pose,
                L_target=L_ref,
                lam2=float(cli.solve_lam2),
                tol_mm=float(cli.solve_tol),
                max_iter=int(cli.solve_max_iter),
                step_scale=float(cli.solve_step_scale),
            )
            t1 = now_ns()

            step_ns_values.append(t1 - t0)
            err_norms.append(err_norm)
            iteration_values.append(float(iterations))
            if not success:
                failures += 1

            row: Dict[str, float] = {"time_s": t}
            row.update({f"Lref{i+1}": L_ref[i] for i in range(6)})
            row.update({f"Lsol{i+1}": L_sol[i] for i in range(6)})
            row.update({
                "xsol": pose[0], "ysol": pose[1], "zsol": pose[2],
                "rollsol": pose[3], "pitchsol": pose[4], "yawsol": pose[5],
                "err_norm_mm": err_norm,
                "iterations": iterations,
                "success": int(success),
                "step_us": ns_to_us(t1 - t0),
            })
            writer.writerow(row)

    errors = np.asarray(err_norms, dtype=float)
    iterations_array = np.asarray(iteration_values, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    metrics: Dict[str, float] = {
        "method": "Iterative-DLS-pose-solve-baseline",
        "rms_error_mm": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_mm": float(np.mean(errors)),
        "max_error_mm": float(np.max(errors)),
        "final_error_mm": float(errors[-1]),
        "max_cond_J": float("nan"),
        "max_rho_Ad": float("nan"),
        "mean_iterations": float(np.mean(iterations_array)),
        "failure_count": int(failures),
    }
    return metrics, step_ms


# -----------------------------------------------------------------------------
# Summary output
# -----------------------------------------------------------------------------
AGGREGATE_FIELDS = [
    "timestamp_utc",
    "run_name",
    "method_key",
    "method",
    "c_vec",
    "target_lengths_mm",
    "pose0",
    "steps",
    "dt_s",
    "submv_T_s",
    "runtime_measured_steps",
    "runtime_warmup_steps",
    "runtime_evaluated_steps",
    "mean_step_ms",
    "median_step_ms",
    "p95_step_ms",
    "p99_step_ms",
    "max_step_ms",
    "rms_error_mm",
    "mean_error_mm",
    "max_error_mm",
    "final_error_mm",
    "max_cond_J",
    "max_rho_Ad",
    "mean_iterations",
    "failure_count",
]


def save_step_times(path: str, step_ms: np.ndarray, warmup_steps: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_index", "step_time_ms", "excluded_as_warmup"])
        for idx, value in enumerate(np.asarray(step_ms, dtype=float)):
            writer.writerow([idx, float(value), int(idx < warmup_steps)])


def write_one_row(path: str, row: Dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in AGGREGATE_FIELDS})


def upsert_aggregate(path: str, row: Dict[str, object]) -> None:
    rows: List[Dict[str, str]] = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    key = (str(row["run_name"]), str(row["method_key"]))
    replaced = False
    for idx, old in enumerate(rows):
        if (old.get("run_name", ""), old.get("method_key", "")) == key:
            rows[idx] = {field: str(row.get(field, "")) for field in AGGREGATE_FIELDS}
            replaced = True
            break

    if not replaced:
        rows.append({field: str(row.get(field, "")) for field in AGGREGATE_FIELDS})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(row: Dict[str, object]) -> None:
    print("\n=== Parallel repeated benchmark summary ===")
    print(f"  run name:        {row['run_name']}")
    print(f"  method:          {row['method']}")
    if row["method_key"] == "ksmp-dls":
        print(f"  C diagonal:      [{row['c_vec']}]")
    print(f"  target lengths:  [{row['target_lengths_mm']}]")
    print(f"  steps/dt/T:      {row['steps']} / {row['dt_s']} s / {row['submv_T_s']} s")
    print(f"  warm-up removed: {row['runtime_warmup_steps']} steps")
    print(f"  mean step time:  {float(row['mean_step_ms']):.6f} ms")
    print(f"  p95  step time:  {float(row['p95_step_ms']):.6f} ms")
    print(f"  RMS length dev:  {float(row['rms_error_mm']):.6g} mm")
    print(f"  final length dev:{float(row['final_error_mm']):.6g} mm")
    if np.isfinite(float(row["mean_iterations"])):
        print(f"  mean solve iters:{float(row['mean_iterations']):.3f}")
        print(f"  solve failures:  {int(row['failure_count'])}")
    print("===========================================\n")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Repeated benchmark for parallel-robot KS-MP and baselines."
    )
    ap.add_argument(
        "--method",
        choices=["ksmp-dls", "leg-pid", "pose-solve"],
        required=True,
        help="Run exactly one method per invocation.",
    )
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--out-dir", "--outdir", dest="out_dir", type=str, default="baseline_results/parallel")
    ap.add_argument("--timing-warmup-steps", type=int, default=10)
    ap.add_argument("--no-print-summary", action="store_true")

    target_group = ap.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-lengths",
        type=float,
        nargs=6,
        default=[1249.900, 1255.800, 1329.800, 1351.200, 1330.900, 1303.300],
        help="Target leg lengths [mm].",
    )
    target_group.add_argument(
        "--target-pose",
        type=float,
        nargs=6,
        help="Target pose [x,y,z,roll,pitch,yaw], converted by the same body schema.",
    )

    ap.add_argument("--pose0", type=float, nargs=6, default=list(pp.POSE0_DEFAULT))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=6.0)

    # KS-MP-DLS settings.
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--kp-vec", type=float, nargs=6, dest="kp_vec")
    ap.add_argument("--dp", type=float, default=0.2)
    ap.add_argument("--dp-vec", type=float, nargs=6, dest="dp_vec")
    ap.add_argument("--lam2", type=float, default=1e-4)
    ap.add_argument("--bq-diag", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0])
    ap.add_argument(
        "--c-vec",
        type=float,
        nargs="+",
        default=None,
        help="KS-MP participation diagonal: one value or six values.",
    )

    # Leg-length PID-style tracking baseline.
    ap.add_argument("--leg-kp", type=float, default=10.0)
    ap.add_argument("--leg-kp-vec", type=float, nargs=6, dest="leg_kp_vec")
    ap.add_argument("--leg-dp", type=float, default=0.0)
    ap.add_argument("--leg-dp-vec", type=float, nargs=6, dest="leg_dp_vec")
    ap.add_argument("--leg-ki", type=float, default=0.0)
    ap.add_argument("--leg-ki-vec", type=float, nargs=6, dest="leg_ki_vec")
    ap.add_argument("--leg-integral-clip", type=float, default=0.0)
    ap.add_argument("--leg-lam2", type=float, default=1e-4)

    # Iterative pose solve.
    ap.add_argument("--solve-lam2", type=float, default=1e-4)
    ap.add_argument("--solve-tol", type=float, default=1e-3)
    ap.add_argument("--solve-max-iter", type=int, default=80)
    ap.add_argument("--solve-step-scale", type=float, default=1.0)
    return ap


def main() -> None:
    cli = build_arg_parser().parse_args()

    if cli.steps <= 0:
        raise ValueError("--steps must be positive")
    if cli.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if cli.submv_T <= 0.0:
        raise ValueError("--submv-T must be positive")
    if cli.lam2 < 0.0 or cli.leg_lam2 < 0.0 or cli.solve_lam2 < 0.0:
        raise ValueError("DLS regularisation values must be non-negative")
    if cli.solve_tol < 0.0:
        raise ValueError("--solve-tol must be non-negative")
    if cli.solve_max_iter <= 0:
        raise ValueError("--solve-max-iter must be positive")
    if cli.timing_warmup_steps < 0:
        raise ValueError("--timing-warmup-steps must be non-negative")

    os.makedirs(cli.out_dir, exist_ok=True)

    if cli.target_pose is not None:
        target_lengths = pp.lengths_from_pose(np.asarray(cli.target_pose, dtype=float))
    else:
        target_lengths = np.asarray(cli.target_lengths, dtype=float)

    run_file = sanitise_filename(cli.run_name)
    log_path = os.path.join(cli.out_dir, f"{run_file}_log.csv")
    step_path = os.path.join(cli.out_dir, f"{run_file}_step_times.csv")
    summary_path = os.path.join(cli.out_dir, f"{run_file}_summary.csv")
    aggregate_path = os.path.join(cli.out_dir, "timing_runs.csv")

    if cli.method == "ksmp-dls":
        metrics, step_ms = run_ksmp_dls(cli, target_lengths, log_path)
    elif cli.method == "leg-pid":
        metrics, step_ms = run_leg_pid(cli, target_lengths, log_path)
    else:
        metrics, step_ms = run_pose_solve(cli, target_lengths, log_path)

    timing = timing_statistics(step_ms, cli.timing_warmup_steps)
    c_vec = resolve_c_vector(cli) if cli.method == "ksmp-dls" else np.full(6, np.nan)

    row: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": cli.run_name,
        "method_key": cli.method,
        "method": metrics["method"],
        "c_vec": serialise_vector(c_vec),
        "target_lengths_mm": serialise_vector(target_lengths),
        "pose0": serialise_vector(cli.pose0),
        "steps": int(cli.steps),
        "dt_s": float(cli.dt),
        "submv_T_s": float(cli.submv_T),
        **timing,
        "rms_error_mm": metrics["rms_error_mm"],
        "mean_error_mm": metrics["mean_error_mm"],
        "max_error_mm": metrics["max_error_mm"],
        "final_error_mm": metrics["final_error_mm"],
        "max_cond_J": metrics["max_cond_J"],
        "max_rho_Ad": metrics["max_rho_Ad"],
        "mean_iterations": metrics["mean_iterations"],
        "failure_count": metrics["failure_count"],
    }

    save_step_times(step_path, step_ms, int(cli.timing_warmup_steps))
    write_one_row(summary_path, row)
    upsert_aggregate(aggregate_path, row)

    if not cli.no_print_summary:
        print_summary(row)
    print(f"Saved run log:       {os.path.abspath(log_path)}")
    print(f"Saved step timings:  {os.path.abspath(step_path)}")
    print(f"Saved run summary:   {os.path.abspath(summary_path)}")
    print(f"Updated aggregate:   {os.path.abspath(aggregate_path)}")


if __name__ == "__main__":
    main()
