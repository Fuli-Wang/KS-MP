#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeated benchmark for the serial-robot KS-MP implementation.

The script compares three algorithmic methods under the same UR10e body
schema and minimum-jerk Cartesian reference:

1. KS-MP with damped least-squares mapping;
2. conventional task-space DLS reference tracking;
3. iterative DLS inverse kinematics solved at each reference point.

Run one method per invocation so that execution order can be interleaved
across repeated trials. Reference generation and diagnostic-only calculations
are excluded from the timed controller section where applicable.

This benchmark reports algorithmic, model-based quantities rather than
physical robot measurements.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

# Import the serial-robot implementation from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SERIAL_ROBOT_DIR = REPO_ROOT / "serial_robot"
if str(SERIAL_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(SERIAL_ROBOT_DIR))

import pmp_serial as ps
from dh_fk import UR10e_DH, T_UR10e_TOOL_TO_RG2FT_TCP


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------
def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(x: float) -> float:
    return float(x) / 1e6


def ns_to_us(x: float) -> float:
    return float(x) / 1e3


def sanitize_name(value: str) -> str:
    """Return a filesystem-safe run name."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value))
    return cleaned.strip("._") or "run"


def timing_eval_values(step_ms: np.ndarray, warmup_steps: int) -> tuple[np.ndarray, int]:
    """Remove run-start warm-up samples from timing statistics."""
    values = np.asarray(step_ms, dtype=float)
    if values.size == 0:
        raise ValueError("No timing samples were collected.")
    warmup = min(max(int(warmup_steps), 0), max(values.size - 1, 0))
    return values[warmup:], warmup


def safe_cond_from_svd(s: np.ndarray, eps: float = 1e-12) -> float:
    if s.size == 0:
        return np.nan
    smax = float(np.max(s))
    smin = float(np.min(s))
    if smin <= eps:
        return np.inf
    return smax / smin


def dls_pinv(J: np.ndarray, lam2: float) -> np.ndarray:
    """Return J^T (J J^T + lambda^2 I)^-1."""
    JT = J.T
    regularised = J @ JT + float(lam2) * np.eye(J.shape[0])
    return JT @ np.linalg.solve(
        regularised,
        np.eye(regularised.shape[0]),
    )


def make_diag(values, n: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        v = np.full(n, float(v[0]))
    if v.size != n:
        raise ValueError(f"Expected {n} values, got {v.size}")
    return np.diag(v)


def spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def compute_local_stability_metrics(
    J: np.ndarray,
    Kp: np.ndarray,
    Dp: np.ndarray,
    C: np.ndarray,
    BQ: np.ndarray,
    dt: float,
    lam2: float,
    use_jt: bool = False,
) -> Dict[str, float]:
    """
    Compute the local effective task-space map H and spectral radius of the
    companion matrix used in the local discrete-time stability discussion.

    Error dynamics approximation:
        dx_{k+1} = (I - dt H K - H D) dx_k + H D dx_{k-1}

    For DLS:
        M = J_lambda^dagger, H = J (I-BQ) C M
    For JT ablation:
        M = J^T, H = J (I-BQ) C M
    """
    task_dim = J.shape[0]
    I_task = np.eye(task_dim)

    if use_jt:
        M = J.T
    else:
        M = dls_pinv(J, lam2=lam2)

    H = J @ (np.eye(J.shape[1]) - BQ) @ C @ M

    A_top_left = I_task - dt * (H @ Kp) - (H @ Dp)
    A_top_right = H @ Dp
    A_bottom_left = I_task
    A_bottom_right = np.zeros_like(I_task)
    A_d = np.block([[A_top_left, A_top_right],
                    [A_bottom_left, A_bottom_right]])

    # Modal gains of H are useful for reporting. H may be non-symmetric, so
    # report eigenvalue magnitudes as a diagnostic, and spectral radius of A_d.
    H_eigs = np.linalg.eigvals(H)
    H_abs = np.abs(H_eigs)
    return {
        "rho_Ad": spectral_radius(A_d),
        "H_eig_abs_min": float(np.min(H_abs)),
        "H_eig_abs_max": float(np.max(H_abs)),
        "H_cond_svd": safe_cond_from_svd(np.linalg.svd(H, compute_uv=False)),
    }


# -----------------------------------------------------------------------------
# Shared controller configuration
# -----------------------------------------------------------------------------
def build_ps_args(cli: argparse.Namespace) -> SimpleNamespace:
    """Create an args object compatible with pmp_serial.core_step."""
    return SimpleNamespace(
        dt=float(cli.dt),
        submv_T=float(cli.submv_T),
        steps=int(cli.steps),
        traj="minjerk",
        target=list(cli.target),
        q0=list(cli.q0),
        kp1=float(cli.kp1),
        kp1_xy=None,
        kp1_z=None,
        kp1_x=None,
        kp1_y=None,
        kp1_z_axis=None,
        dp_x=float(cli.dp),
        dp_y=float(cli.dp),
        dp_z=float(cli.dp),
        use_jt=False,          # reported implementation: DLS
        lam2=float(cli.lam2),
        bq_diag=list(cli.bq_diag),
        c_vec=(None if cli.c_vec is None else list(cli.c_vec)),
        timing_warmup_steps=int(cli.timing_warmup_steps),
        # Unused by minjerk but kept for compatibility.
        osc_ax=0.0,
        osc_ay=55.0,
        osc_cycles=1.0,
        osc_use_drift=False,
        osc_lx=150.0,
        peg_dz=0.15,
        peg_osc_amp=1.0,
        peg_osc_freq=1.0,
        peg_xy_mode="x0",
        peg_xy_tfrac=0.2,
    )


def reference_minjerk(x0: np.ndarray, target: np.ndarray, t: float, T: float) -> np.ndarray:
    s = ps.min_jerk_s(t, T)
    return x0 + s * (target - x0)


def min_jerk_sdot(t: float, T: float) -> float:
    """Time derivative of the scalar minimum-jerk phase s(t)."""
    if T <= 0.0:
        return 0.0
    tau = np.clip(t / T, 0.0, 1.0)
    return (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T


def reference_minjerk_with_velocity(
    x0: np.ndarray,
    target: np.ndarray,
    t: float,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return minimum-jerk reference position and analytic reference velocity."""
    s = ps.min_jerk_s(t, T)
    sdot = min_jerk_sdot(t, T)
    delta = target - x0
    return x0 + s * delta, sdot * delta


def get_gains_and_filters(args: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Kp = np.diag([args.kp1, args.kp1, args.kp1]).astype(float)
    Dp = np.diag([args.dp_x, args.dp_y, args.dp_z]).astype(float)

    if args.c_vec is None:
        C_vec = np.ones(6, dtype=float)
    else:
        C_vec = np.asarray(args.c_vec, dtype=float)
    C = make_diag(C_vec, 6)
    BQ = make_diag(args.bq_diag, 6)
    return Kp, Dp, C, BQ


# -----------------------------------------------------------------------------
# KS-MP DLS benchmark
# -----------------------------------------------------------------------------
def run_ksmp_dls_benchmark(args: SimpleNamespace, target_xyz: np.ndarray, out_csv: str) -> Dict[str, float]:
    q = np.asarray(args.q0, dtype=float)
    x0, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    x0 = np.asarray(x0, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)

    x_prev = x0.copy()
    x_ref_prev = x0.copy()
    Kp, Dp, C, BQ = get_gains_and_filters(args)

    fieldnames = [
        "time_s",
        "x", "y", "z", "xref", "yref", "zref",
        "err_x", "err_y", "err_z", "err_norm_mm",
        "Fx", "Fy", "Fz",
        "qdot1", "qdot2", "qdot3", "qdot4", "qdot5", "qdot6",
        "q1", "q2", "q3", "q4", "q5", "q6",
        "sigma_min_J", "sigma_max_J", "cond_J",
        "rho_Ad", "H_eig_abs_min", "H_eig_abs_max", "H_cond_svd",
        "step_us", "fk_us", "jac_us", "dls_us", "diag_us",
    ]

    err_norms = []
    step_ns_values = []
    cond_values = []
    rho_values = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(args.steps):
            t = k * args.dt
            x_ref = reference_minjerk(x0, target_xyz, t, args.submv_T)

            t0 = now_ns()

            # FK timing
            tf0 = now_ns()
            x, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
            x = np.asarray(x, dtype=float)
            tf1 = now_ns()

            # Virtual task-space interaction signal
            xdot_now = (x - x_prev) / args.dt
            xdot_ref = (x_ref - x_ref_prev) / args.dt
            e = x_ref - x
            F_task = Kp @ e + Dp @ (xdot_ref - xdot_now)

            # Jacobian timing
            tj0 = now_ns()
            J = UR10e_DH.jacobian_xyz_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP).astype(float)
            tj1 = now_ns()

            # DLS timing
            td0 = now_ns()
            Jpinv = dls_pinv(J, lam2=args.lam2)
            tau = Jpinv @ F_task
            qdot = (np.eye(6) - BQ) @ (C @ tau)
            td1 = now_ns()

            q_next = q + qdot * args.dt
            # Stop the controller-update timer before diagnostic-only calculations.
            t1 = now_ns()

            # Diagnostics are logged but excluded from controller runtime.
            tg0 = now_ns()
            svals = np.linalg.svd(J, compute_uv=False)
            condJ = safe_cond_from_svd(svals)
            diag = compute_local_stability_metrics(
                J=J, Kp=Kp, Dp=Dp, C=C, BQ=BQ, dt=args.dt, lam2=args.lam2, use_jt=False
            )
            tg1 = now_ns()

            err_norm = float(np.linalg.norm(e))
            err_norms.append(err_norm)
            step_ns_values.append(t1 - t0)
            cond_values.append(condJ)
            rho_values.append(diag["rho_Ad"])

            writer.writerow({
                "time_s": t,
                "x": x[0], "y": x[1], "z": x[2],
                "xref": x_ref[0], "yref": x_ref[1], "zref": x_ref[2],
                "err_x": e[0], "err_y": e[1], "err_z": e[2], "err_norm_mm": err_norm,
                "Fx": F_task[0], "Fy": F_task[1], "Fz": F_task[2],
                "qdot1": qdot[0], "qdot2": qdot[1], "qdot3": qdot[2],
                "qdot4": qdot[3], "qdot5": qdot[4], "qdot6": qdot[5],
                "q1": q_next[0], "q2": q_next[1], "q3": q_next[2],
                "q4": q_next[3], "q5": q_next[4], "q6": q_next[5],
                "sigma_min_J": float(np.min(svals)),
                "sigma_max_J": float(np.max(svals)),
                "cond_J": condJ,
                "rho_Ad": diag["rho_Ad"],
                "H_eig_abs_min": diag["H_eig_abs_min"],
                "H_eig_abs_max": diag["H_eig_abs_max"],
                "H_cond_svd": diag["H_cond_svd"],
                "step_us": ns_to_us(t1 - t0),
                "fk_us": ns_to_us(tf1 - tf0),
                "jac_us": ns_to_us(tj1 - tj0),
                "dls_us": ns_to_us(td1 - td0),
                "diag_us": ns_to_us(tg1 - tg0),
            })

            q = q_next
            x_prev = x.copy()
            x_ref_prev = x_ref.copy()

    err_norms = np.asarray(err_norms, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    eval_step_ms, warmup = timing_eval_values(step_ms, args.timing_warmup_steps)
    return {
        "method": "KS-MP-DLS",
        "mean_step_ms": float(np.mean(eval_step_ms)),
        "p95_step_ms": float(np.percentile(eval_step_ms, 95)),
        "timing_warmup_steps": int(warmup),
        "timing_measured_steps": int(eval_step_ms.size),
        "rms_error_mm": float(np.sqrt(np.mean(err_norms ** 2))),
        "mean_error_mm": float(np.mean(err_norms)),
        "max_error_mm": float(np.max(err_norms)),
        "final_error_mm": float(err_norms[-1]),
        "max_cond_J": float(np.nanmax(cond_values)),
        "max_rho_Ad": float(np.max(rho_values)),
        "mean_iterations": np.nan,
        "failure_count": 0,
    }



# -----------------------------------------------------------------------------
# Task-space DLS tracking baseline
# -----------------------------------------------------------------------------
def run_task_space_tracking_baseline(cli: argparse.Namespace, target_xyz: np.ndarray, out_csv: str) -> Dict[str, float]:
    """
    Standard kinematic task-space tracking baseline.

    This baseline uses the same minimum-jerk reference as KS-MP-DLS, but treats
    it as an imposed reference trajectory to be tracked:

        qdot = J_lambda^dagger [ xdot_ref
                                 + Kx (x_ref - x)
                                 + Dx (xdot_ref - xdot) ]

    It is intended as a conventional minimum-jerk task-space tracking baseline,
    not as an EDA/PMP relaxation update. It therefore has no C-weighted
    participation structure and no KS-MP companion stability matrix A_d.
    """
    q = np.asarray(cli.q0, dtype=float)
    x0, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    x0 = np.asarray(x0, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)

    Kx = np.eye(3) * float(cli.track_kp)
    Dx = np.eye(3) * float(cli.track_kd)

    x_prev = x0.copy()

    fieldnames = [
        "time_s",
        "x", "y", "z", "xref", "yref", "zref",
        "xdot_ref", "ydot_ref", "zdot_ref",
        "err_x", "err_y", "err_z", "err_norm_mm",
        "v_cmd_x", "v_cmd_y", "v_cmd_z",
        "qdot1", "qdot2", "qdot3", "qdot4", "qdot5", "qdot6",
        "q1", "q2", "q3", "q4", "q5", "q6",
        "sigma_min_J", "sigma_max_J", "cond_J",
        "step_us", "fk_us", "jac_us", "dls_us",
    ]

    err_norms = []
    step_ns_values = []
    cond_values = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(int(cli.steps)):
            t = k * float(cli.dt)
            x_ref, xdot_ref = reference_minjerk_with_velocity(x0, target_xyz, t, float(cli.submv_T))

            t0 = now_ns()

            # FK timing.
            tf0 = now_ns()
            x, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
            x = np.asarray(x, dtype=float)
            tf1 = now_ns()

            xdot_now = (x - x_prev) / float(cli.dt)
            e = x_ref - x

            # Standard task-space reference-tracking command.
            v_cmd = xdot_ref + Kx @ e + Dx @ (xdot_ref - xdot_now)

            # Jacobian timing.
            tj0 = now_ns()
            J = UR10e_DH.jacobian_xyz_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP).astype(float)
            tj1 = now_ns()

            # DLS mapping timing.
            td0 = now_ns()
            qdot = dls_pinv(J, lam2=float(cli.track_lam2)) @ v_cmd
            td1 = now_ns()

            q_next = q + qdot * float(cli.dt)
            t1 = now_ns()

            svals = np.linalg.svd(J, compute_uv=False)
            condJ = safe_cond_from_svd(svals)
            err_norm = float(np.linalg.norm(e))

            err_norms.append(err_norm)
            step_ns_values.append(t1 - t0)
            cond_values.append(condJ)

            writer.writerow({
                "time_s": t,
                "x": x[0], "y": x[1], "z": x[2],
                "xref": x_ref[0], "yref": x_ref[1], "zref": x_ref[2],
                "xdot_ref": xdot_ref[0], "ydot_ref": xdot_ref[1], "zdot_ref": xdot_ref[2],
                "err_x": e[0], "err_y": e[1], "err_z": e[2], "err_norm_mm": err_norm,
                "v_cmd_x": v_cmd[0], "v_cmd_y": v_cmd[1], "v_cmd_z": v_cmd[2],
                "qdot1": qdot[0], "qdot2": qdot[1], "qdot3": qdot[2],
                "qdot4": qdot[3], "qdot5": qdot[4], "qdot6": qdot[5],
                "q1": q_next[0], "q2": q_next[1], "q3": q_next[2],
                "q4": q_next[3], "q5": q_next[4], "q6": q_next[5],
                "sigma_min_J": float(np.min(svals)),
                "sigma_max_J": float(np.max(svals)),
                "cond_J": condJ,
                "step_us": ns_to_us(t1 - t0),
                "fk_us": ns_to_us(tf1 - tf0),
                "jac_us": ns_to_us(tj1 - tj0),
                "dls_us": ns_to_us(td1 - td0),
            })

            q = q_next
            x_prev = x.copy()

    err_norms = np.asarray(err_norms, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    eval_step_ms, warmup = timing_eval_values(step_ms, cli.timing_warmup_steps)
    return {
        "method": "Task-space-DLS-tracking-baseline",
        "mean_step_ms": float(np.mean(eval_step_ms)),
        "p95_step_ms": float(np.percentile(eval_step_ms, 95)),
        "timing_warmup_steps": int(warmup),
        "timing_measured_steps": int(eval_step_ms.size),
        "rms_error_mm": float(np.sqrt(np.mean(err_norms ** 2))),
        "mean_error_mm": float(np.mean(err_norms)),
        "max_error_mm": float(np.max(err_norms)),
        "final_error_mm": float(err_norms[-1]),
        "max_cond_J": float(np.nanmax(cond_values)),
        "max_rho_Ad": np.nan,
        "mean_iterations": np.nan,
        "failure_count": 0,
    }


# -----------------------------------------------------------------------------
# Iterative DLS IK baseline
# -----------------------------------------------------------------------------
def solve_ik_dls_single(
    q_seed: np.ndarray,
    x_target: np.ndarray,
    lam2: float,
    tol_mm: float,
    max_iter: int,
    step_scale: float,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    """Solve one Cartesian reference point using iterative DLS IK."""
    q = np.asarray(q_seed, dtype=float).copy()
    x_target = np.asarray(x_target, dtype=float)

    success = False
    x = None
    err_norm = np.inf

    for it in range(1, max_iter + 1):
        x, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
        x = np.asarray(x, dtype=float)
        e = x_target - x
        err_norm = float(np.linalg.norm(e))
        if err_norm <= tol_mm:
            success = True
            break
        J = UR10e_DH.jacobian_xyz_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP).astype(float)
        dq = dls_pinv(J, lam2=lam2) @ e
        q = q + float(step_scale) * dq

    # One final FK for the returned residual if loop ended after update.
    x, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    x = np.asarray(x, dtype=float)
    err_norm = float(np.linalg.norm(x_target - x))
    if err_norm <= tol_mm:
        success = True
    return q, x, err_norm, it, success


def run_iterative_ik_baseline(cli: argparse.Namespace, target_xyz: np.ndarray, out_csv: str) -> Dict[str, float]:
    q = np.asarray(cli.q0, dtype=float)
    x0, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    x0 = np.asarray(x0, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)

    fieldnames = [
        "time_s",
        "xref", "yref", "zref",
        "xsol", "ysol", "zsol",
        "err_norm_mm", "iterations", "success", "step_us",
    ]

    step_ns_values = []
    err_norms = []
    iter_values = []
    failures = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(int(cli.steps)):
            t = k * float(cli.dt)
            x_ref = reference_minjerk(x0, target_xyz, t, float(cli.submv_T))

            t0 = now_ns()
            q, x_sol, err_norm, iters, success = solve_ik_dls_single(
                q_seed=q,
                x_target=x_ref,
                lam2=float(cli.ik_lam2),
                tol_mm=float(cli.ik_tol),
                max_iter=int(cli.ik_max_iter),
                step_scale=float(cli.ik_step_scale),
            )
            t1 = now_ns()

            if not success:
                failures += 1
            step_ns_values.append(t1 - t0)
            err_norms.append(err_norm)
            iter_values.append(iters)

            writer.writerow({
                "time_s": t,
                "xref": x_ref[0], "yref": x_ref[1], "zref": x_ref[2],
                "xsol": x_sol[0], "ysol": x_sol[1], "zsol": x_sol[2],
                "err_norm_mm": err_norm,
                "iterations": iters,
                "success": int(success),
                "step_us": ns_to_us(t1 - t0),
            })

    err_norms = np.asarray(err_norms, dtype=float)
    step_ms = np.asarray([ns_to_ms(v) for v in step_ns_values], dtype=float)
    eval_step_ms, warmup = timing_eval_values(step_ms, cli.timing_warmup_steps)
    iter_values = np.asarray(iter_values, dtype=float)
    return {
        "method": "Iterative-DLS-IK-baseline",
        "mean_step_ms": float(np.mean(eval_step_ms)),
        "p95_step_ms": float(np.percentile(eval_step_ms, 95)),
        "timing_warmup_steps": int(warmup),
        "timing_measured_steps": int(eval_step_ms.size),
        "rms_error_mm": float(np.sqrt(np.mean(err_norms ** 2))),
        "mean_error_mm": float(np.mean(err_norms)),
        "max_error_mm": float(np.max(err_norms)),
        "final_error_mm": float(err_norms[-1]),
        "max_cond_J": np.nan,
        "max_rho_Ad": np.nan,
        "mean_iterations": float(np.mean(iter_values)),
        "failure_count": int(failures),
    }


# -----------------------------------------------------------------------------
# Summary output
# -----------------------------------------------------------------------------
SUMMARY_FIELDS = [
    "run_name", "method_key", "method",
    "target_x", "target_y", "target_z",
    "steps", "dt", "submv_T",
    "timing_warmup_steps", "timing_measured_steps",
    "mean_step_ms", "p95_step_ms",
    "rms_error_mm", "mean_error_mm", "max_error_mm", "final_error_mm",
    "max_cond_J", "max_rho_Ad", "mean_iterations", "failure_count",
    "lam2", "track_kp", "track_kd", "track_lam2",
    "ik_lam2", "ik_tol", "ik_max_iter", "ik_step_scale",
]


def write_summary(summary_path: str, rows: list[Dict[str, float]]) -> None:
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_or_replace_runs(path: str, new_rows: list[Dict[str, float]]) -> None:
    """Append runs, replacing an existing row with the same run_name and method_key."""
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            existing = list(csv.DictReader(f))

    replacement_keys = {(str(r["run_name"]), str(r["method_key"])) for r in new_rows}
    existing = [
        r for r in existing
        if (str(r.get("run_name", "")), str(r.get("method_key", ""))) not in replacement_keys
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(new_rows)


def print_summary(rows: list[Dict[str, float]]) -> None:
    print("\n=== Serial repeated benchmark summary ===")
    for r in rows:
        print(f"  run name:        {r['run_name']}")
        print(f"  method:          {r['method']}")
        print(f"  target:          [{r['target_x']}, {r['target_y']}, {r['target_z']}]")
        print(f"  steps/dt/T:      {r['steps']} / {r['dt']} s / {r['submv_T']} s")
        print(f"  warm-up removed: {r['timing_warmup_steps']} steps")
        print(f"  mean step time:  {r['mean_step_ms']:.6f} ms")
        print(f"  p95  step time:  {r['p95_step_ms']:.6f} ms")
        print(f"  RMS error:       {r['rms_error_mm']:.6g} mm")
        print(f"  final error:     {r['final_error_mm']:.6g} mm")
        if not np.isnan(r.get("mean_iterations", np.nan)):
            print(f"  mean IK iters:   {r['mean_iterations']:.3f}")
            print(f"  IK failures:     {r['failure_count']}")
        print("-")
    print("===========================================\n")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        choices=["ksmp-dls", "tracking-dls", "iterative-ik"],
        required=True,
        help="Run one method per invocation for interleaved repeated timing.",
    )
    ap.add_argument("--run-name", type=str, default="benchmark_run")
    ap.add_argument("--timing-warmup-steps", type=int, default=10)
    ap.add_argument("--target", type=float, nargs=3, default=[-491.73, 181.25, 119.76])
    ap.add_argument("--q0", type=float, nargs=6, default=[180.0, -85.0, -100.0, 0.0, 90.0, 357.58])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=6.0)
    ap.add_argument("--kp1", type=float, default=100.0)
    ap.add_argument("--dp", type=float, default=0.2)
    ap.add_argument("--lam2", type=float, default=1e-4)
    ap.add_argument("--bq-diag", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0])
    ap.add_argument("--c-vec", type=float, nargs=6, default=None)
    ap.add_argument("--out-dir", "--outdir", dest="outdir", type=str, default="baseline_results/serial")

    # Standard task-space DLS tracking baseline settings.
    ap.add_argument("--track-kp", type=float, default=10.0,
                    help="Task-space proportional feedback gain for the tracking baseline [1/s].")
    ap.add_argument("--track-kd", type=float, default=0.0,
                    help="Task-space velocity feedback gain for the tracking baseline [dimensionless].")
    ap.add_argument("--track-lam2", type=float, default=1e-4,
                    help="DLS damping for the task-space tracking baseline.")

    # Iterative IK baseline settings.
    ap.add_argument("--ik-lam2", type=float, default=1e-4)
    ap.add_argument("--ik-tol", type=float, default=1e-3, help="IK tolerance in mm")
    ap.add_argument("--ik-max-iter", type=int, default=80)
    ap.add_argument("--ik-step-scale", type=float, default=1.0)
    return ap


def enrich_row(row: Dict[str, float], cli: argparse.Namespace, method_key: str) -> Dict[str, float]:
    target = [float(v) for v in cli.target]
    enriched = dict(row)
    enriched.update({
        "run_name": str(cli.run_name),
        "method_key": method_key,
        "target_x": target[0],
        "target_y": target[1],
        "target_z": target[2],
        "steps": int(cli.steps),
        "dt": float(cli.dt),
        "submv_T": float(cli.submv_T),
        "lam2": float(cli.lam2),
        "track_kp": float(cli.track_kp),
        "track_kd": float(cli.track_kd),
        "track_lam2": float(cli.track_lam2),
        "ik_lam2": float(cli.ik_lam2),
        "ik_tol": float(cli.ik_tol),
        "ik_max_iter": int(cli.ik_max_iter),
        "ik_step_scale": float(cli.ik_step_scale),
    })
    return enriched


def main() -> None:
    parser = build_arg_parser()
    cli = parser.parse_args()

    if cli.steps <= 0:
        raise ValueError("--steps must be positive")
    if cli.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if cli.submv_T <= 0.0:
        raise ValueError("--submv-T must be positive")
    if cli.lam2 < 0.0 or cli.track_lam2 < 0.0 or cli.ik_lam2 < 0.0:
        raise ValueError("DLS regularisation values must be non-negative")
    if cli.ik_tol < 0.0:
        raise ValueError("--ik-tol must be non-negative")
    if cli.ik_max_iter <= 0:
        raise ValueError("--ik-max-iter must be positive")
    if cli.timing_warmup_steps < 0:
        raise ValueError("--timing-warmup-steps must be non-negative")

    os.makedirs(cli.outdir, exist_ok=True)

    target = np.asarray(cli.target, dtype=float)
    ps_args = build_ps_args(cli)
    safe_run = sanitize_name(cli.run_name)

    method_keys = [cli.method]

    rows: list[Dict[str, float]] = []
    for method_key in method_keys:
        if method_key == "ksmp-dls":
            log_path = os.path.join(cli.outdir, f"{safe_run}_ksmp_dls_log.csv")
            row = run_ksmp_dls_benchmark(ps_args, target, log_path)
        elif method_key == "tracking-dls":
            log_path = os.path.join(cli.outdir, f"{safe_run}_tracking_dls_log.csv")
            row = run_task_space_tracking_baseline(cli, target, log_path)
        elif method_key == "iterative-ik":
            log_path = os.path.join(cli.outdir, f"{safe_run}_iterative_ik_log.csv")
            row = run_iterative_ik_baseline(cli, target, log_path)
        else:
            raise ValueError(f"Unsupported method: {method_key}")
        rows.append(enrich_row(row, cli, method_key))

    run_summary = os.path.join(cli.outdir, f"{safe_run}_summary.csv")
    all_runs = os.path.join(cli.outdir, "timing_runs.csv")
    write_summary(run_summary, rows)
    append_or_replace_runs(all_runs, rows)
    print_summary(rows)
    print(f"Saved run summary to: {os.path.abspath(run_summary)}")
    print(f"Updated all-runs table: {os.path.abspath(all_runs)}")


if __name__ == "__main__":
    main()
