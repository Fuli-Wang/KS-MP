#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-MP controller for a soft arm using a learned body schema.

The learned model maps internal coordinates q to Cartesian tip position and
provides the corresponding Jacobian through automatic differentiation. The
controller supports minimum-jerk, VTGS, and oscillatory references; DLS and
Jacobian-transpose mappings; diagonal participation and damping matrices;
runtime diagnostics; and trajectory export for SoRoSim replay.

Units:
- Tip position: mm
- Time: s
- Internal-coordinate units follow the training dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import numpy as np
import torch
from dataclasses import dataclass
from train_softarm import SoftArmPosNet


# =========================
# Default parameters
# =========================
ITERATION_DEFAULT = 1500
DT_DEFAULT        = 0.004
SUBMV_T_DEFAULT   = 6.0
TRAJ_DEF          = "minjerk"   # "vtgs" or "minjerk" or "osc"
KP1_DEFAULT       = 100.0
LAM2_DEFAULT      = 1e-4

# Scalar defaults are expanded to the model input dimension when required.
BQ_DIAG_DEFAULT   = [0.0]
C_DEFAULT         = None  # None selects full participation.

# Shared parameters used by the VTGS helper functions
RAMP_KONSTANT = DT_DEFAULT
t_dur         = SUBMV_T_DEFAULT


# =========================
# Reference-generation helpers
# =========================
def min_jerk_s(t: float, T: float) -> float:
    """Scalar min-jerk phase s(t) in [0, 1] for t in [0, T]."""
    if T <= 0.0:
        return 1.0
    tau = np.clip(t / T, 0.0, 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


def GammaDisc(t_idx: int) -> float:
    """Return the discrete VTGS kernel at one controller step."""
    t = t_idx * RAMP_KONSTANT
    if t_dur <= 0.0:
        return 0.0
    tau = np.clip(t / t_dur, 0.0, 1.0)
    return 30.0 * (tau**2) * ((1.0 - tau)**2)


def Gamma_IntDisc(Gam_arr: np.ndarray, t_idx: int) -> float:
    """
    Composite Simpson-like integration over discrete Gamma array up to index t_idx.
    step = dt = RAMP_KONSTANT, returns integral(Gamma * dt).
    """
    if t_idx <= 0:
        return 0.0
    h = RAMP_KONSTANT
    n = t_idx
    if n % 2 == 1:  # Simpson integration requires an even interval count
        n -= 1
    if n < 2:
        return Gam_arr[:t_idx + 1].sum() * h
    s = Gam_arr[0] + Gam_arr[n]
    s += 4.0 * Gam_arr[1:n:2].sum()
    s += 2.0 * Gam_arr[2:n - 1:2].sum()
    return s * (h / 3.0)


# =========================
# Differential mapping
# =========================
def dls_pinv(J: np.ndarray, lam2: float = 1e-4) -> np.ndarray:
    """
    Damped least-squares pseudoinverse:
        J^+ = Jᵀ (J Jᵀ + λ² I)⁻¹
    """
    JT = J.T
    JJt = J @ JT
    lam2I = lam2 * np.eye(JJt.shape[0])
    return JT @ np.linalg.inv(JJt + lam2I)


def _jacobian_condition_numbers(model: SoftArmPosNet,
                                q_traj: np.ndarray,
                                device,
                                stride: int = 10) -> np.ndarray:
    """
    Compute condition numbers of the learned Jacobian along a reaching trajectory.

    This is a controller-internal diagnostic based on the learned map
    J_theta(q)=d x / d q. It is not a physical ground-truth Jacobian check.
    """
    if q_traj.size == 0:
        return np.asarray([], dtype=float)

    stride = max(1, int(stride))
    q_sample = q_traj[::stride]

    q_t = torch.from_numpy(q_sample).float().to(device)
    model.eval()

    # Do not use torch.no_grad(), because model.jacobian() uses autograd.
    J = model.jacobian(q_t).detach().cpu().numpy()  # [B, 3, ndof]

    conds = []
    for Ji in J:
        s = np.linalg.svd(Ji, compute_uv=False)
        smax = float(np.max(s))
        smin = float(np.min(s))
        conds.append(smax / max(smin, 1e-12))

    return np.asarray(conds, dtype=float)


def _save_metrics_csv(metrics: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                writer.writerow([k, ",".join([str(float(x)) for x in v])])
            else:
                writer.writerow([k, v])


def summarize_reaching_metrics(logs_arr: np.ndarray,
                               ndof: int,
                               model: SoftArmPosNet,
                               args,
                               step_times_ms: np.ndarray | None = None) -> dict:
    """
    Summarise controller-internal reaching performance.

    logs_arr columns:
        time,
        x,y,z,
        x_ref,y_ref,z_ref,
        Fx,Fy,Fz,
        qdot1...qdotN,
        q1...qN

    The tip-position errors are computed using the controller-internal learned
    body-schema prediction x, not a SoRoSim replay/ground-truth tip.
    """
    device = next(model.parameters()).device

    t = logs_arr[:, 0]
    x = logs_arr[:, 1:4]
    x_ref = logs_arr[:, 4:7]
    F_task = logs_arr[:, 7:10]
    qdot = logs_arr[:, 10:10 + ndof]
    q_traj = logs_arr[:, 10 + ndof:10 + 2 * ndof]

    err = x_ref - x
    err_norm = np.linalg.norm(err, axis=1)

    per_axis_rms = np.sqrt(np.mean(err ** 2, axis=0))
    F_norm = np.linalg.norm(F_task, axis=1)
    qdot_norm = np.linalg.norm(qdot, axis=1)

    conds = _jacobian_condition_numbers(
        model=model,
        q_traj=q_traj,
        device=device,
        stride=getattr(args, "jac_stride", 10),
    )

    metrics = {
        "steps": int(len(t)),
        "dt": float(args.dt),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "mapping": "JT" if args.use_jt else "DLS",
        "lam2": float(args.lam2),

        # Euclidean reaching deviation, in mm
        "rms_euclidean_mm": float(np.sqrt(np.mean(err_norm ** 2))),
        "mean_euclidean_mm": float(np.mean(err_norm)),
        "final_euclidean_mm": float(err_norm[-1]),
        "p95_euclidean_mm": float(np.percentile(err_norm, 95)),
        "max_euclidean_mm": float(np.max(err_norm)),

        # Per-axis deviation, in mm
        "rms_x_mm": float(per_axis_rms[0]),
        "rms_y_mm": float(per_axis_rms[1]),
        "rms_z_mm": float(per_axis_rms[2]),
        "final_x_error_mm": float(err[-1, 0]),
        "final_y_error_mm": float(err[-1, 1]),
        "final_z_error_mm": float(err[-1, 2]),

        # Final state
        "x0_mm": x[0].tolist(),
        "x_final_mm": x[-1].tolist(),
        "x_ref_final_mm": x_ref[-1].tolist(),
        "q_final": q_traj[-1].tolist(),

        # Control/update magnitude
        "F_norm_max": float(np.max(F_norm)),
        "F_norm_p95": float(np.percentile(F_norm, 95)),
        "qdot_norm_max": float(np.max(qdot_norm)),
        "qdot_norm_p95": float(np.percentile(qdot_norm, 95)),
    }

    if conds.size > 0:
        metrics.update({
            "jacobian_cond_mean": float(np.mean(conds)),
            "jacobian_cond_p95": float(np.percentile(conds, 95)),
            "jacobian_cond_max": float(np.max(conds)),
            "jacobian_cond_num_samples": int(len(conds)),
            "jacobian_cond_stride": int(getattr(args, "jac_stride", 10)),
        })

    # Runtime/computational-cost metrics for the controller loop.
    # These are wall-clock measurements of one control step, including reference
    # generation, learned prediction, learned Jacobian evaluation, DLS/JT mapping,
    # Euler update, and logging overhead. If CUDA is used, the loop synchronises
    # before stopping the timer.
    if step_times_ms is not None:
        step_times_ms = np.asarray(step_times_ms, dtype=float)
        if step_times_ms.size > 0:
            warmup = int(getattr(args, "timing_warmup_steps", 10))
            warmup = min(max(warmup, 0), max(step_times_ms.size - 1, 0))
            eval_times = step_times_ms[warmup:]
            metrics.update({
                "runtime_device": str(next(model.parameters()).device),
                "runtime_measured_steps": int(step_times_ms.size),
                "runtime_warmup_steps": int(warmup),
                "runtime_total_s": float(np.sum(step_times_ms) / 1000.0),
                "runtime_step_mean_ms": float(np.mean(eval_times)),
                "runtime_step_median_ms": float(np.median(eval_times)),
                "runtime_step_p95_ms": float(np.percentile(eval_times, 95)),
                "runtime_step_p99_ms": float(np.percentile(eval_times, 99)),
                "runtime_step_max_ms": float(np.max(eval_times)),
                "runtime_mean_frequency_hz": float(1000.0 / max(np.mean(eval_times), 1e-12)),
            })

    return metrics


# =========================
# Controller parameters
# =========================
@dataclass
class SoftArgs:
    dt: float           = DT_DEFAULT
    submv_T: float      = SUBMV_T_DEFAULT
    steps: int          = ITERATION_DEFAULT
    traj: str           = TRAJ_DEF
    # Oscillation-reference parameters
    osc_ax: float       = 50.0
    osc_ay: float       = 50.0
    osc_cycles: float   = 1.0
    osc_use_drift: bool = False
    osc_lx: float       = 0.0

    # Cartesian gains
    kp1: float | None      = KP1_DEFAULT
    kp1_xy: float | None   = None
    kp1_z: float | None    = None
    kp1_x: float | None    = None
    kp1_y: float | None    = None
    kp1_z_axis: float | None = None

    dp_x: float = 0.2
    dp_y: float = 0.2
    dp_z: float = 0.2

    use_jt: bool  = False
    lam2: float   = LAM2_DEFAULT

    # Internal-coordinate participation and damping
    bq_diag: list | None = None
    c_vec: list | None   = None        # PMP-style participation/compliance C

    # Initial internal coordinates
    q0: np.ndarray | None     = None

    # Cartesian target in mm
    target: np.ndarray | None = None
    # Optional translation from the learned-model frame to the control frame
    x_offset: np.ndarray | None = None

    # Output and diagnostics
    out_dir: str = "soft_reaching_results"
    run_name: str = "soft_reach"
    jac_stride: int = 10
    timing_warmup_steps: int = 10


# =========================
# KS-MP control step
# =========================
def core_step_soft(
    q: np.ndarray,
    x_ref: np.ndarray,
    x_prev: np.ndarray,
    x_ref_prev: np.ndarray,
    args: SoftArgs,
    model: SoftArmPosNet,
):
    """
    Advance the internal coordinates by one KS-MP update.

    Parameters
    ----------
    q : ndarray, shape (n_dof,)
        Current internal coordinates.
    x_ref : ndarray, shape (3,)
        Cartesian tip reference in mm.
    x_prev, x_ref_prev : ndarray, shape (3,)
        Previous predicted and reference positions.
    args : SoftArgs
        Controller parameters.
    model : SoftArmPosNet
        Learned body schema and analytical Jacobian.
    """
    ndof = q.shape[0]

    # Learned forward map and Jacobian
    device = next(model.parameters()).device
    q_t = torch.from_numpy(q).float().unsqueeze(0).to(device)

    # Predicted Cartesian tip position
    x_phys_t = model.predict(q_t)            # [1,3] in mm
    x_phys = x_phys_t[0].detach().cpu().numpy()

    # Learned differential map, shape [3, n_dof]
    J_t = model.jacobian(q_t)                # [1,3,ndof]
    J = J_t[0].detach().cpu().numpy()        # (3, ndof)

    # Apply an optional control-frame translation.
    if getattr(args, "x_offset", None) is not None:
        x = x_phys + np.asarray(args.x_offset, dtype=float)
    else:
        x = x_phys

    # Cartesian virtual command
    # Cartesian gain matrix
    if args.kp1 is not None:
        Kp = np.diag([args.kp1, args.kp1, args.kp1])
    else:
        # Allow either planar/vertical gains or individual axis gains.
        if args.kp1_x is not None or args.kp1_y is not None or args.kp1_z_axis is not None:
            kx = args.kp1_x if args.kp1_x is not None else (args.kp1_xy or KP1_DEFAULT)
            ky = args.kp1_y if args.kp1_y is not None else (args.kp1_xy or KP1_DEFAULT)
            kz = args.kp1_z_axis if args.kp1_z_axis is not None else (args.kp1_z or kx)
        else:
            kxy = args.kp1_xy if args.kp1_xy is not None else KP1_DEFAULT
            kz  = args.kp1_z  if args.kp1_z  is not None else kxy
            kx = ky = kxy
        Kp = np.diag([kx, ky, kz])

    e = (x_ref - x)  # [mm]

    # Cartesian damping matrix
    Dp = np.diag([args.dp_x, args.dp_y, args.dp_z])

    # Backward-difference velocity estimates
    dt = args.dt
    xdot_now = (x - x_prev) / dt
    xdot_ref = (x_ref - x_ref_prev) / dt
    v_err = xdot_ref - xdot_now

    # Cartesian virtual command
    F_task = Kp @ e + Dp @ v_err   # [3,]

    # Differential mapping and participation matrix
    # Map the Cartesian command into internal coordinates.
    if args.use_jt:
        # Jacobian-transpose mapping
        tau = J.T @ F_task          # (ndof,)
    else:
        # Damped least-squares mapping
        Jpinv = dls_pinv(J, lam2=args.lam2)
        tau = Jpinv @ F_task        # (ndof,)

    # Internal-coordinate participation values
    if args.c_vec is None:
        # Full participation by default
        C_vec = np.ones(ndof, dtype=float)
    else:
        C_vec = np.asarray(args.c_vec, dtype=float)
        if C_vec.size == 1:
            C_vec = np.full(ndof, C_vec[0], dtype=float)
        elif C_vec.size != ndof:
            raise ValueError(f"c_vec must contain 1 or {ndof} values; received {C_vec.size}")

    C_mat = np.diag(C_vec)

    # Apply the participation matrix.
    qdot_task = C_mat @ tau    # (ndof,)

    # Internal-coordinate damping
    if args.bq_diag is None:
        BQ_vec = np.zeros(ndof, dtype=float)
    else:
        BQ_vec = np.asarray(args.bq_diag, dtype=float)
        if BQ_vec.size == 1:
            BQ_vec = np.full(ndof, BQ_vec[0], dtype=float)
        elif BQ_vec.size != ndof:
            raise ValueError(f"bq_diag must contain 1 or {ndof} values; received {BQ_vec.size}")
    BQ = np.diag(BQ_vec)

    # Apply diagonal damping.
    qdot = (np.eye(ndof) - BQ) @ qdot_task

    # Explicit Euler integration
    q_next = q + qdot * dt

    return q_next, x, F_task, qdot


# =========================
# Controller execution
# =========================
def run_controller(args: SoftArgs, model: SoftArmPosNet, target_xyz: np.ndarray):
    """Execute a reference-driven KS-MP rollout using the learned body schema."""
    global RAMP_KONSTANT, t_dur
    RAMP_KONSTANT = args.dt
    t_dur         = args.submv_T

    device = next(model.parameters()).device

    # Initial internal coordinates
    if args.q0 is None:
        # Use the origin of the learned coordinate space when q0 is omitted.
        ndof = model.q_mean.numel()
        q0 = np.zeros(ndof, dtype=float)
        use_default_x0 = True
    else:
        q0 = np.asarray(args.q0, dtype=float)
        ndof = q0.size
        use_default_x0 = False

    args.q0 = q0

    # Initial learned-model tip position
    with torch.no_grad():
         x0_phys_t = model.predict(torch.from_numpy(q0).float().unsqueeze(0).to(device))
    x0_phys = x0_phys_t[0].cpu().numpy()   # (3,)

    if use_default_x0:
        x0_ctrl_desired = np.array([900.0, 0.0, 0.0], dtype=float)
        x_offset = x0_ctrl_desired - x0_phys
    else:
        x_offset = np.zeros(3, dtype=float)

    args.x_offset = x_offset
    x0 = x0_phys + x_offset
    # Previous states used for velocity estimation
    x_prev = x0.copy()
    x_ref_prev = x0.copy()

    # VTGS integration buffers
    GamX = np.zeros(args.steps, dtype=float)
    GamY = np.zeros(args.steps, dtype=float)
    GamZ = np.zeros(args.steps, dtype=float)

    logs = []
    step_times_ms = []

    for t_idx in range(args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_t0 = time.perf_counter()

        t = t_idx * args.dt

        # Generate the Cartesian reference
        if args.traj == "vtgs":
            Gam = GammaDisc(t_idx)
            GamX[t_idx] = Gam * (target_xyz[0] - x0[0])
            GamY[t_idx] = Gam * (target_xyz[1] - x0[1])
            GamZ[t_idx] = Gam * (target_xyz[2] - x0[2])
            xr = x0[0] + Gamma_IntDisc(GamX, t_idx)
            yr = x0[1] + Gamma_IntDisc(GamY, t_idx)
            zr = x0[2] + Gamma_IntDisc(GamZ, t_idx)

        elif args.traj == "minjerk":
            s = min_jerk_s(t, args.submv_T)
            xr = x0[0] + s * (target_xyz[0] - x0[0])
            yr = x0[1] + s * (target_xyz[1] - x0[1])
            zr = x0[2] + s * (target_xyz[2] - x0[2])

        elif args.traj == "osc":
            # Oscillatory reference
            tau = min_jerk_s(t, t_dur);#tau = 0.0 if t_dur <= 0.0 else np.clip(t / t_dur, 0.0, 1.0)
            theta = 2.0 * np.pi * args.osc_cycles * tau

            if not args.osc_use_drift:
                # Closed oscillation around the initial tip position
                x_center = x0[0]
                y_center = x0[1]
                z_center = x0[2] - args.osc_ay

                xr = x_center
                yr = y_center + args.osc_ax * np.sin(theta)
                zr = z_center + args.osc_ay * np.cos(theta)
            else:
                xr = x0[0]
                yr = x0[1] + args.osc_lx * tau
                zr = x0[2] + args.osc_ay * np.sin(theta)
        else:
            raise ValueError(f"Unknown traj type: {args.traj}")

        x_ref = np.array([xr, yr, zr], dtype=float)

        # Apply one KS-MP update
        q0, x, F_task, qdot = core_step_soft(q0, x_ref, x_prev, x_ref_prev, args, model)

        x_prev = x.copy()
        x_ref_prev = x_ref.copy()

        logs.append([
            t,
            *x.tolist(),
            *x_ref.tolist(),
            *F_task.tolist(),
            *qdot.tolist(),
            *q0.tolist(),
        ])

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_t1 = time.perf_counter()
        step_times_ms.append((step_t1 - step_t0) * 1000.0)

    logs_arr = np.asarray(logs, dtype=float)
    step_times_ms = np.asarray(step_times_ms, dtype=float)

    # Output directory
    out_dir = getattr(args, "out_dir", "soft_reaching_results")
    run_name = getattr(args, "run_name", "soft_reach")
    os.makedirs(out_dir, exist_ok=True)

    if logs_arr.size == 0:
        np.savetxt(os.path.join(out_dir, f"{run_name}_results.txt"),
                   np.zeros((0, 9), dtype=float), fmt="%f")
        np.savetxt(os.path.join(out_dir, f"{run_name}_results_head.csv"),
                   np.zeros((0, 1), dtype=float), fmt="%f", header="", comments="")
        return q0, x0

    # Extract logged state arrays
    ndof = int(model.q_mean.numel())
    x_cols = logs_arr[:, 1:4]
    x_ref_cols = logs_arr[:, 4:7]
    qdot_cols = logs_arr[:, 10:10 + ndof]
    q_cols = logs_arr[:, 10 + ndof:10 + 2 * ndof]

    # Save the internal-coordinate trajectory for SoRoSim replay
    np.savetxt(
        os.path.join(out_dir, f"{run_name}_q_traj.csv"),
        q_cols,
        delimiter=",",
        fmt="%.10f",
        header=",".join([f"q{i+1}" for i in range(ndof)]),
        comments="",
    )

    # Save learned-model predictions, references, and errors
    # Columns: time, predicted position, reference, and reference error
    x_pred_ref = np.hstack([logs_arr[:, [0]], x_cols, x_ref_cols, x_ref_cols - x_cols])
    np.savetxt(
        os.path.join(out_dir, f"{run_name}_x_pred_ref.csv"),
        x_pred_ref,
        delimiter=",",
        fmt="%.10f",
        header="time,x,y,z,x_ref,y_ref,z_ref,ex,ey,ez",
        comments="",
    )

    # Save the complete controller log
    header = (
        "time,x,y,z,x_ref,y_ref,z_ref,Fx,Fy,Fz," +
        ",".join([f"qdot{i+1}" for i in range(ndof)]) + "," +
        ",".join([f"q{i+1}" for i in range(ndof)])
    )
    np.savetxt(
        os.path.join(out_dir, f"{run_name}_results_head.csv"),
        logs_arr,
        fmt="%.10f",
        delimiter=",",
        header=header,
        comments="",
    )

    # Compact output: internal coordinates followed by predicted position.
    basic = np.hstack([q_cols, x_cols])
    np.savetxt(os.path.join(out_dir, f"{run_name}_results.txt"), basic, fmt="%.10f")

    # Compute and save rollout metrics
    metrics = summarize_reaching_metrics(logs_arr, ndof, model, args, step_times_ms=step_times_ms)

    metrics_json_path = os.path.join(out_dir, f"{run_name}_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metrics_csv_path = os.path.join(out_dir, f"{run_name}_metrics.csv")
    _save_metrics_csv(metrics, metrics_csv_path)

    # Print the rollout summary
    print("\nReaching metrics (controller-internal prediction)")
    print("-----------------------------------------------")
    print(f"RMS Euclidean error      : {metrics['rms_euclidean_mm']:.6f} mm")
    print(f"Final Euclidean error    : {metrics['final_euclidean_mm']:.6f} mm")
    print(f"P95 Euclidean error      : {metrics['p95_euclidean_mm']:.6f} mm")
    print(f"Max Euclidean error      : {metrics['max_euclidean_mm']:.6f} mm")
    print(f"RMS x/y/z error          : {metrics['rms_x_mm']:.6f}, "
          f"{metrics['rms_y_mm']:.6f}, {metrics['rms_z_mm']:.6f} mm")
    if "jacobian_cond_mean" in metrics:
        print(f"Jacobian cond mean / p95 : {metrics['jacobian_cond_mean']:.6f}, "
              f"{metrics['jacobian_cond_p95']:.6f}")
    if "runtime_step_mean_ms" in metrics:
        print(f"Runtime mean / p95       : {metrics['runtime_step_mean_ms']:.6f}, "
              f"{metrics['runtime_step_p95_ms']:.6f} ms/step")
        print(f"Runtime max / freq       : {metrics['runtime_step_max_ms']:.6f} ms, "
              f"{metrics['runtime_mean_frequency_hz']:.2f} Hz")
    print(f"Saved metrics to         : {metrics_json_path}")
    print(f"Saved trajectory to      : {os.path.join(out_dir, f'{run_name}_x_pred_ref.csv')}")
    print(f"Saved q trajectory to    : {os.path.join(out_dir, f'{run_name}_q_traj.csv')}")

    return q0, x


# =========================
# Command-line interface
# =========================
def _build_arg_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str,
                    default="softarm_pos_net.pth",
                    help="path to trained SoftArmPosNet checkpoint")

    ap.add_argument("--target", type=float, nargs=3, default=[790.0, -11.0, -364.0],
                    help="Cartesian goal position in the controller frame [mm]")
    ap.add_argument("--q0", type=float, nargs="+", default=None,
                    help="initial internal coordinates q0 (length=ndof); default all zeros")

    ap.add_argument("--steps", type=int, default=ITERATION_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=SUBMV_T_DEFAULT)

    ap.add_argument("--traj", choices=["vtgs", "minjerk", "osc"], default=TRAJ_DEF)

    # Oscillation-reference parameters
    ap.add_argument("--osc-ax", type=float, default=50.0,
                    help="oscillation amplitude along x [mm]")
    ap.add_argument("--osc-ay", type=float, default=50.0,
                    help="oscillation amplitude along y [mm]")
    ap.add_argument("--osc-cycles", type=float, default=1.0,
                    help="number of oscillation cycles over one primitive duration")
    ap.add_argument("--osc-use-drift", action="store_true",
                    help="add linear drift to form an open oscillatory trajectory")
    ap.add_argument("--osc-lx", type=float, default=150.0,
                    help="drift length [mm] when --osc-use-drift is enabled")

    # Cartesian gains
    ap.add_argument("--kp1", type=float, default=KP1_DEFAULT,
                    help="isotropic task gain (overrides others)")
    ap.add_argument("--kp1-xy", type=float, default=None, help="planar gain for x/y")
    ap.add_argument("--kp1-z", type=float, default=None, help="vertical gain for z")
    ap.add_argument("--kp1-x", type=float, default=None, help="per-axis gain: x")
    ap.add_argument("--kp1-y", type=float, default=None, help="per-axis gain: y")
    ap.add_argument("--kp1-z-axis", dest="kp1_z_axis", type=float, default=None,
                    help="per-axis gain: z")

    # Cartesian damping
    ap.add_argument("--dp-x", type=float, default=0.2, help="task-space damping along x")
    ap.add_argument("--dp-y", type=float, default=0.2, help="task-space damping along y")
    ap.add_argument("--dp-z", type=float, default=0.2, help="task-space damping along z")

    ap.add_argument("--use-jt", action="store_true", default=False,
                    help="use J^T mapping instead of DLS J^+")

    ap.add_argument("--lam2", type=float, default=LAM2_DEFAULT,
                    help="DLS regularization lambda^2")

    # Internal-coordinate damping and participation; one value is broadcast.
    ap.add_argument("--bq-diag", type=float, nargs="+", default=BQ_DIAG_DEFAULT,
                    help="joint damping diagonal entries (1 or ndof values)")
    ap.add_argument("--c-vec", type=float, nargs="+", default=C_DEFAULT,
                    help="participation values (1 or n_dof entries); 0=disabled, 1=full")

    # Output and diagnostics
    ap.add_argument("--out-dir", type=str, default="soft_reaching_results",
                    help="directory for reaching logs and metrics")
    ap.add_argument("--run-name", type=str, default="soft_reach",
                    help="prefix for saved reaching logs and metrics")
    ap.add_argument("--jac-stride", type=int, default=10,
                    help="compute learned-Jacobian condition number every N controller steps")
    ap.add_argument("--timing-warmup-steps", type=int, default=10,
                    help="number of initial control steps excluded from runtime statistics")

    return ap


def main():
    parser = _build_arg_parser()
    args_ns = parser.parse_args()
    #args_ns.osc_use_drift = False  # Preserve the reported non-drift default.

    # Assemble controller parameters
    sargs = SoftArgs(
        dt=args_ns.dt,
        submv_T=args_ns.submv_T,
        steps=args_ns.steps,
        traj=args_ns.traj,
        osc_ax=args_ns.osc_ax,
        osc_ay=args_ns.osc_ay,
        osc_cycles=args_ns.osc_cycles,
        osc_use_drift=args_ns.osc_use_drift,
        osc_lx=args_ns.osc_lx,
        kp1=args_ns.kp1,
        kp1_xy=args_ns.kp1_xy,
        kp1_z=args_ns.kp1_z,
        kp1_x=args_ns.kp1_x,
        kp1_y=args_ns.kp1_y,
        kp1_z_axis=args_ns.kp1_z_axis,
        dp_x=args_ns.dp_x,
        dp_y=args_ns.dp_y,
        dp_z=args_ns.dp_z,
        use_jt=args_ns.use_jt,
        lam2=args_ns.lam2,
        bq_diag=args_ns.bq_diag,
        c_vec=args_ns.c_vec,
        q0=np.array(args_ns.q0, dtype=float) if args_ns.q0 is not None else None,
        target=np.array(args_ns.target, dtype=float),
        out_dir=args_ns.out_dir,
        run_name=args_ns.run_name,
        jac_stride=args_ns.jac_stride,
        timing_warmup_steps=args_ns.timing_warmup_steps,
    )

    # Load the learned body schema
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")
    model = load_softarm_model(args_ns.ckpt, device=device)
    print(f"[info] model loaded from {args_ns.ckpt}")

    # Execute the controller
    q_final, x_final = run_controller(sargs, model, sargs.target)

    print("[info] goal/primitive execution done. q_final =", q_final)
    print("[info] x_final(mm) =", x_final)


def load_softarm_model(ckpt_path: str, device: str = "cpu") -> SoftArmPosNet:
    """Helper to load SoftArmPosNet with weights_only=False."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = ckpt["q_mean"].shape[0]
    out_dim = ckpt["x_mean"].shape[0]

    model = SoftArmPosNet(in_dim=in_dim, out_dim=out_dim, width=256, depth=4)
    model.load_state_dict(ckpt["model"])
    model.set_normalization(ckpt["q_mean"], ckpt["q_std"], ckpt["x_mean"], ckpt["x_std"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    main()
