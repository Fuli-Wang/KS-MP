#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate goal-directed commands for the physical continuum-robot example.

A learned body schema predicts planar tip position from four motor commands.
Because the collected data lie on a two-dimensional actuation manifold, the
controller operates in reduced coordinates

    u = [middle, distal]

with the platform-specific mapping

    q4 = [ID1, ID4, ID5, ID6] = A u.

The KS-MP update uses the learned Jacobian, a Cartesian virtual command,
DLS or Jacobian-transpose mapping, diagonal participation and damping, and
explicit Euler integration. The script exports model-based trajectories and
downsampled commands for hardware replay; it does not communicate directly
with the robot.

Example
-------
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --target-sample 154
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from train_physical_continuum import (
    PhysicalContinuumPosNet,
    read_delimited_table,
)


A_REDUCED_TO_Q4 = np.array(
    [
        [1.0, 0.0],   # ID1 = m
        [1.0, 0.0],   # ID4 = m
        [0.0, -1.0],  # ID5 = -d
        [0.0, 1.0],   # ID6 = +d
    ],
    dtype=np.float64,
)


def min_jerk(t: float, duration: float) -> float:
    if duration <= 0.0:
        return 1.0
    tau = float(np.clip(t / duration, 0.0, 1.0))
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def dls_pinv(jacobian: np.ndarray, lam2: float) -> np.ndarray:
    """
    Return the damped least-squares pseudoinverse

        J^+ = J^T (J J^T + lam2 I)^-1.

    ``lam2`` is the regularisation term added directly to ``J J^T``.
    """
    jj_t = jacobian @ jacobian.T
    regularised = jj_t + lam2 * np.eye(jj_t.shape[0])
    return jacobian.T @ np.linalg.solve(
        regularised,
        np.eye(regularised.shape[0]),
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> PhysicalContinuumPosNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    widths = checkpoint.get("widths", [64, 64, 32])
    model = PhysicalContinuumPosNet(
        in_dim=int(checkpoint.get("in_dim", 4)),
        out_dim=int(checkpoint.get("out_dim", 2)),
        widths=widths,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    input_order = checkpoint.get(
        "input_order",
        ["ID1", "ID4", "ID5", "ID6"],
    )
    if input_order != ["ID1", "ID4", "ID5", "ID6"]:
        raise ValueError(
            f"Unexpected checkpoint input order: {input_order}"
        )

    return model


def predict_xy(
    model: PhysicalContinuumPosNet,
    q4: np.ndarray,
    anchor_bias: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    q_tensor = torch.as_tensor(
        q4,
        dtype=torch.float32,
        device=device,
    )
    pred = model.predict(q_tensor).squeeze(0).cpu().numpy()
    return pred.astype(np.float64) + anchor_bias


def model_jacobian_q4(
    model: PhysicalContinuumPosNet,
    q4: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    q_tensor = torch.as_tensor(
        q4,
        dtype=torch.float32,
        device=device,
    )
    jac = model.jacobian(q_tensor).squeeze(0).detach().cpu().numpy()
    return jac.astype(np.float64)


def sample_target_from_files(
    zed_path: Path,
    schedule_path: Path,
    sample_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    zed_rows = read_delimited_table(zed_path)
    schedule_rows = read_delimited_table(schedule_path)

    zed_by_id = {
        int(float(row["sample_id"])): row for row in zed_rows
    }
    schedule_by_id = {
        int(float(row["sample_id"])): row for row in schedule_rows
    }

    if sample_id not in zed_by_id:
        raise ValueError(f"Sample {sample_id} is absent from ZED data.")
    if sample_id not in schedule_by_id:
        raise ValueError(f"Sample {sample_id} is absent from schedule.")

    z = zed_by_id[sample_id]
    s = schedule_by_id[sample_id]

    target_xy = np.array(
        [float(z["x_mm"]), float(z["y_mm"])],
        dtype=np.float64,
    )
    target_q4 = np.array(
        [
            float(s["ID1"]),
            float(s["ID4"]),
            float(s["ID5"]),
            float(s["ID6"]),
        ],
        dtype=np.float64,
    )
    return target_xy, target_q4


def reduced_from_q4(q4: np.ndarray) -> np.ndarray:
    """Project a four-motor command onto the sampled actuation manifold."""
    middle = 0.5 * (q4[0] + q4[1])
    distal = 0.5 * (q4[3] - q4[2])
    return np.array([middle, distal], dtype=np.float64)


def q4_from_reduced(u: np.ndarray) -> np.ndarray:
    return A_REDUCED_TO_Q4 @ u



def infer_goal_u_grid(
    model: PhysicalContinuumPosNet,
    target_xy: np.ndarray,
    anchor_bias: np.ndarray,
    device: torch.device,
    middle_bounds: Tuple[float, float],
    distal_bounds: Tuple[float, float],
    grid_step: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Resolve an explicit planar target to a bounded reduced-coordinate goal.
    """
    middle_values = np.arange(
        middle_bounds[0], middle_bounds[1] + 0.5 * grid_step, grid_step
    )
    distal_values = np.arange(
        distal_bounds[0], distal_bounds[1] + 0.5 * grid_step, grid_step
    )
    mm, dd = np.meshgrid(middle_values, distal_values, indexing="ij")
    u_grid = np.column_stack([mm.ravel(), dd.ravel()])
    q_grid = u_grid @ A_REDUCED_TO_Q4.T

    with torch.no_grad():
        q_tensor = torch.as_tensor(
            q_grid, dtype=torch.float32, device=device
        )
        xy_grid = model.predict(q_tensor).cpu().numpy().astype(np.float64)
        xy_grid += anchor_bias.reshape(1, 2)

    errors = np.linalg.norm(xy_grid - target_xy.reshape(1, 2), axis=1)
    best = int(np.argmin(errors))
    return u_grid[best], xy_grid[best], float(errors[best])


def two_stage_actuation_reference(
    t: float,
    u_home: np.ndarray,
    u_goal: np.ndarray,
    distal_duration: float,
    middle_duration: float,
) -> Tuple[np.ndarray, str, float]:
    """
    Generate a two-stage actuation-space reference.

    The distal coordinate moves first, followed by the middle coordinate.
    The returned middle-stage phase is used for smooth endpoint correction.
    """
    u_ref = u_home.copy()

    if t <= distal_duration:
        s_distal = min_jerk(t, distal_duration)
        u_ref[1] = u_home[1] + s_distal * (u_goal[1] - u_home[1])
        return u_ref, "distal", 0.0

    u_ref[1] = u_goal[1]
    t_middle = t - distal_duration

    if t_middle <= middle_duration:
        s_middle = min_jerk(t_middle, middle_duration)
        u_ref[0] = u_home[0] + s_middle * (u_goal[0] - u_home[0])
        return u_ref, "middle", s_middle

    u_ref[:] = u_goal
    return u_ref, "settle", 1.0


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even when CUDA is available.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/physical_continuum_body_schema.pt"
        ),
    )
    parser.add_argument(
        "--zed",
        type=Path,
        default=Path("example_data/zed_measurements.csv"),
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("example_data/sampling_schedule.csv"),
    )
    parser.add_argument(
        "--target-sample",
        type=int,
        default=154,
        help=(
            "Use the measured XY and motor state of this sample as the "
            "target. Default: 154."
        ),
    )
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        metavar=("X_MM", "Y_MM"),
        default=None,
        help="Explicit target position. Overrides --target-sample.",
    )
    parser.add_argument(
        "--home-xy",
        nargs=2,
        type=float,
        default=[52.0, -39.0],
        metavar=("X_MM", "Y_MM"),
        help="Measured HOME tip position. Default: 52 -39.",
    )
    parser.add_argument(
        "--home-q4",
        nargs=4,
        type=float,
        default=[0.0, 0.0, 0.0, 0.0],
        metavar=("ID1", "ID4", "ID5", "ID6"),
    )
    parser.add_argument(
        "--distal-duration",
        type=float,
        default=3.0,
        help="Duration of distal/tip-section motion in seconds.",
    )
    parser.add_argument(
        "--middle-duration",
        type=float,
        default=4.0,
        help="Duration of middle/nearer active-section motion in seconds.",
    )
    parser.add_argument(
        "--goal-grid-step",
        type=float,
        default=2.0,
        help="Grid spacing in degrees when resolving an explicit XY target.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.004,
        help="Controller integration step in seconds. Default: 0.004.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=100.0,
        help="Default isotropic task-space stiffness gain. Default: 100.",
    )
    parser.add_argument(
        "--kp-x",
        type=float,
        default=None,
        help="Optional X-axis task-space stiffness override.",
    )
    parser.add_argument(
        "--kp-y",
        type=float,
        default=None,
        help="Optional Y-axis task-space stiffness override.",
    )
    parser.add_argument(
        "--dp",
        "--kd",
        dest="dp",
        type=float,
        default=0.2,
        help=(
            "Default isotropic task-space damping gain. Default: 0.2. "
            "The legacy option name --kd is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--dp-x",
        type=float,
        default=None,
        help="Optional X-axis task-space damping override.",
    )
    parser.add_argument(
        "--dp-y",
        type=float,
        default=None,
        help="Optional Y-axis task-space damping override.",
    )
    parser.add_argument(
        "--c-middle",
        type=float,
        default=1.0,
        help="Participation/compliance weight for the middle coordinate.",
    )
    parser.add_argument(
        "--c-distal",
        type=float,
        default=1.0,
        help="Participation/compliance weight for the distal coordinate.",
    )
    parser.add_argument(
        "--bq-middle",
        type=float,
        default=0.0,
        help=(
            "Reduced-coordinate damping for the middle coordinate in "
            "(I-B_Q)u_dot. Default: 0."
        ),
    )
    parser.add_argument(
        "--bq-distal",
        type=float,
        default=0.0,
        help=(
            "Reduced-coordinate damping for the distal coordinate in "
            "(I-B_Q)u_dot. Default: 0."
        ),
    )
    parser.add_argument(
        "--use-jt",
        action="store_true",
        help=(
            "Use J^T F_task instead of the default DLS J^+ F_task mapping. "
            "DLS is used by default."
        ),
    )
    parser.add_argument(
        "--lam2",
        type=float,
        default=1.0e-4,
        help=(
            "DLS regularisation added directly to J J^T. "
            "Default: 1e-4, using the same regularisation convention as the other KS-MP scripts."
        ),
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help=(
            "Legacy compatibility option. If provided, --lam2 is set to "
            "damping^2, preserving the previous continuum-code convention."
        ),
    )
    parser.add_argument(
        "--max-u-rate",
        type=float,
        default=45.0,
        help="Maximum reduced-coordinate speed in deg/s.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help=(
            "Weight of the current saturated rate in the first-order "
            "command smoother; must lie in (0, 1]. Default: 1.0 "
            "(no additional smoothing, matching the analysed update law)."
        ),
    )
    parser.add_argument(
        "--middle-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--middle-max",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--distal-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--distal-max",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="Extra time after the reference reaches the target.",
    )
    parser.add_argument(
        "--command-period",
        type=float,
        default=0.20,
        help="Time spacing between saved replay commands.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("physical_reaching_results"),
    )
    parser.add_argument(
        "--timing-warmup-steps",
        type=int,
        default=10,
        help=(
            "Number of initial controller steps excluded from runtime "
            "statistics. Default: 10."
        ),
    )
    args = parser.parse_args()

    # Retain the deprecated --damping alias for compatibility.
    if args.damping is not None:
        if args.damping < 0.0:
            raise ValueError("--damping must be non-negative.")
        args.lam2 = float(args.damping) ** 2

    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.distal_duration <= 0.0:
        raise ValueError("--distal-duration must be positive.")
    if args.middle_duration <= 0.0:
        raise ValueError("--middle-duration must be positive.")
    if args.command_period < args.dt:
        raise ValueError("--command-period must be >= --dt.")
    if args.kp < 0.0 or args.dp < 0.0:
        raise ValueError("--kp and --dp/--kd must be non-negative.")
    for name in ("kp_x", "kp_y", "dp_x", "dp_y"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.c_middle < 0.0 or args.c_distal < 0.0:
        raise ValueError("Participation weights must be non-negative.")
    if not (0.0 <= args.bq_middle <= 1.0):
        raise ValueError("--bq-middle must lie in [0, 1].")
    if not (0.0 <= args.bq_distal <= 1.0):
        raise ValueError("--bq-distal must lie in [0, 1].")
    if args.lam2 < 0.0:
        raise ValueError("--lam2 must be non-negative.")
    if args.timing_warmup_steps < 0:
        raise ValueError("--timing-warmup-steps must be non-negative.")
    if args.max_u_rate <= 0.0:
        raise ValueError("--max-u-rate must be positive.")
    if not (0.0 < args.smoothing <= 1.0):
        raise ValueError("--smoothing must lie in (0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = load_model(args.checkpoint, device)

    home_q4 = np.asarray(args.home_q4, dtype=np.float64)
    home_xy = np.asarray(args.home_xy, dtype=np.float64)

    # Align the learned model with the measured HOME position.
    raw_home_prediction = predict_xy(
        model,
        home_q4,
        anchor_bias=np.zeros(2),
        device=device,
    )
    anchor_bias = home_xy - raw_home_prediction

    if args.target is not None:
        target_xy = np.asarray(args.target, dtype=np.float64)
        target_q4_reference = None
        target_description = "explicit"
        u_goal, resolved_goal_xy, resolution_error = infer_goal_u_grid(
            model=model,
            target_xy=target_xy,
            anchor_bias=anchor_bias,
            device=device,
            middle_bounds=(args.middle_min, args.middle_max),
            distal_bounds=(args.distal_min, args.distal_max),
            grid_step=args.goal_grid_step,
        )
    else:
        target_xy, target_q4_reference = sample_target_from_files(
            args.zed,
            args.schedule,
            args.target_sample,
        )
        target_description = f"sample_{args.target_sample}"
        u_goal = reduced_from_q4(target_q4_reference)
        resolved_goal_xy = predict_xy(
            model,
            q4_from_reduced(u_goal),
            anchor_bias=anchor_bias,
            device=device,
        )
        resolution_error = float(np.linalg.norm(resolved_goal_xy - target_xy))

    u_home = reduced_from_q4(home_q4)
    u = u_home.copy()
    u_dot_previous = np.zeros(2, dtype=np.float64)
    x_previous = home_xy.copy()
    x_ref_previous = home_xy.copy()

    kp_x = args.kp if args.kp_x is None else args.kp_x
    kp_y = args.kp if args.kp_y is None else args.kp_y
    dp_x = args.dp if args.dp_x is None else args.dp_x
    dp_y = args.dp if args.dp_y is None else args.dp_y
    Kp = np.diag([kp_x, kp_y]).astype(np.float64)
    Dp = np.diag([dp_x, dp_y]).astype(np.float64)
    C_u = np.diag([args.c_middle, args.c_distal]).astype(np.float64)
    BQ = np.diag([args.bq_middle, args.bq_distal]).astype(np.float64)
    I2 = np.eye(2, dtype=np.float64)

    reference_duration = args.distal_duration + args.middle_duration
    total_time = reference_duration + args.settle_time
    steps = int(math.ceil(total_time / args.dt)) + 1

    log_rows: List[Dict[str, object]] = []
    step_times_ms: List[float] = []

    for step in range(steps):
        # CUDA kernels are asynchronous. Synchronise at both timing boundaries
        # to measure the actual end-to-end controller-step latency.
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        t = step * args.dt
        q4 = q4_from_reduced(u)

        x_hat = predict_xy(
            model,
            q4,
            anchor_bias=anchor_bias,
            device=device,
        )

        if step == 0:
            x_dot = np.zeros(2, dtype=np.float64)
        else:
            x_dot = (x_hat - x_previous) / args.dt

        u_ref, reference_stage, middle_phase = two_stage_actuation_reference(
            t=t,
            u_home=u_home,
            u_goal=u_goal,
            distal_duration=args.distal_duration,
            middle_duration=args.middle_duration,
        )
        q4_ref = q4_from_reduced(u_ref)
        x_ref_manifold = predict_xy(
            model,
            q4_ref,
            anchor_bias=anchor_bias,
            device=device,
        )

        # Follow the learned manifold and smoothly correct its endpoint
        # during the second stage.
        endpoint_correction = target_xy - resolved_goal_xy
        x_ref = x_ref_manifold + middle_phase * endpoint_correction

        if step == 0:
            x_ref_dot = np.zeros(2, dtype=np.float64)
        else:
            x_ref_dot = (x_ref - x_ref_previous) / args.dt

        task_error = x_ref - x_hat
        velocity_error = x_ref_dot - x_dot

        # Cartesian virtual command:
        #   F_task = Kp (x_ref - x) + Dp (x_ref_dot - x_dot)
        F_task = Kp @ task_error + Dp @ velocity_error

        jac_q4 = model_jacobian_q4(model, q4, device)
        jac_u = jac_q4 @ A_REDUCED_TO_Q4

        singular_values = np.linalg.svd(
            jac_u,
            compute_uv=False,
        )
        condition_number = float(
            singular_values[0] / max(singular_values[-1], 1e-12)
        )

        # Map the Cartesian command to the two admissible actuation coordinates.
        if args.use_jt:
            reduced_drive = jac_u.T @ F_task
        else:
            reduced_drive = dls_pinv(jac_u, args.lam2) @ F_task

        # Apply reduced-coordinate participation and damping.
        u_dot_task = C_u @ reduced_drive
        u_dot_raw = (I2 - BQ) @ u_dot_task
        rate_saturated = bool(
            np.any(np.abs(u_dot_raw) > args.max_u_rate)
        )
        u_dot_saturated = np.clip(
            u_dot_raw,
            -args.max_u_rate,
            args.max_u_rate,
        )

        # Smooth the saturated command rate.
        u_dot = (
            args.smoothing * u_dot_saturated
            + (1.0 - args.smoothing) * u_dot_previous
        )

        u_next = u + args.dt * u_dot
        u_next[0] = np.clip(
            u_next[0],
            args.middle_min,
            args.middle_max,
        )
        u_next[1] = np.clip(
            u_next[1],
            args.distal_min,
            args.distal_max,
        )

        error = target_xy - x_hat
        ref_error = x_ref - x_hat

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - step_start) * 1000.0
        step_times_ms.append(float(step_time_ms))

        log_rows.append(
            {
                "step": step,
                "time_s": round(t, 6),
                "x_hat_mm": float(x_hat[0]),
                "y_hat_mm": float(x_hat[1]),
                "x_ref_mm": float(x_ref[0]),
                "y_ref_mm": float(x_ref[1]),
                "x_ref_manifold_mm": float(x_ref_manifold[0]),
                "y_ref_manifold_mm": float(x_ref_manifold[1]),
                "reference_stage": reference_stage,
                "middle_ref_deg": float(u_ref[0]),
                "distal_ref_deg": float(u_ref[1]),
                "x_target_mm": float(target_xy[0]),
                "y_target_mm": float(target_xy[1]),
                "target_error_mm": float(np.linalg.norm(error)),
                "reference_error_mm": float(np.linalg.norm(ref_error)),
                "middle_u_deg": float(u[0]),
                "distal_u_deg": float(u[1]),
                "x_dot_mm_s": float(x_dot[0]),
                "y_dot_mm_s": float(x_dot[1]),
                "x_ref_dot_mm_s": float(x_ref_dot[0]),
                "y_ref_dot_mm_s": float(x_ref_dot[1]),
                "x_velocity_error_mm_s": float(velocity_error[0]),
                "y_velocity_error_mm_s": float(velocity_error[1]),
                "F_task_x": float(F_task[0]),
                "F_task_y": float(F_task[1]),
                "reduced_drive_middle": float(reduced_drive[0]),
                "reduced_drive_distal": float(reduced_drive[1]),
                "middle_rate_task_deg_s": float(u_dot_task[0]),
                "distal_rate_task_deg_s": float(u_dot_task[1]),
                "middle_rate_raw_deg_s": float(u_dot_raw[0]),
                "distal_rate_raw_deg_s": float(u_dot_raw[1]),
                "middle_rate_saturated_deg_s": float(u_dot_saturated[0]),
                "distal_rate_saturated_deg_s": float(u_dot_saturated[1]),
                "rate_saturated": int(rate_saturated),
                "middle_rate_deg_s": float(u_dot[0]),
                "distal_rate_deg_s": float(u_dot[1]),
                "ID1": float(q4[0]),
                "ID4": float(q4[1]),
                "ID5": float(q4[2]),
                "ID6": float(q4[3]),
                "jacobian_condition": condition_number,
                "jacobian_sigma_min": float(singular_values[-1]),
                "jacobian_sigma_max": float(singular_values[0]),
                "controller_step_time_ms": float(step_time_ms),
            }
        )

        x_previous = x_hat.copy()
        x_ref_previous = x_ref.copy()
        u_dot_previous = u_dot.copy()
        u = u_next

    final = log_rows[-1]
    final_q4 = np.array(
        [
            final["ID1"],
            final["ID4"],
            final["ID5"],
            final["ID6"],
        ],
        dtype=np.float64,
    )

    warmup = min(
        max(int(args.timing_warmup_steps), 0),
        max(len(step_times_ms) - 1, 0),
    )
    timing_eval = np.asarray(step_times_ms[warmup:], dtype=np.float64)
    if timing_eval.size == 0:
        timing_eval = np.asarray(step_times_ms, dtype=np.float64)
        warmup = 0

    dt_ms = 1000.0 * float(args.dt)
    runtime_mean_ms = float(np.mean(timing_eval))
    runtime_median_ms = float(np.median(timing_eval))
    runtime_std_ms = float(np.std(timing_eval))
    runtime_p95_ms = float(np.percentile(timing_eval, 95))
    runtime_p99_ms = float(np.percentile(timing_eval, 99))
    runtime_max_ms = float(np.max(timing_eval))
    runtime_frequency_hz = float(1000.0 / max(runtime_mean_ms, 1e-12))
    deadline_miss_fraction = float(np.mean(timing_eval > dt_ms))

    metrics = {
        "target_description": target_description,
        "target_xy_mm": target_xy.tolist(),
        "resolved_goal_u": u_goal.tolist(),
        "resolved_goal_model_xy_mm": resolved_goal_xy.tolist(),
        "goal_resolution_error_mm": resolution_error,
        "reference_order": ["distal", "middle"],
        "controller_form": (
            "F_task = Kp*(x_ref-x) + Dp*(x_ref_dot-x_dot); "
            "u_dot = (I-B_Q) C_u mapping(F_task)"
        ),
        "mapping": "JT" if args.use_jt else "DLS",
        "kp_xy": [float(kp_x), float(kp_y)],
        "dp_xy": [float(dp_x), float(dp_y)],
        "c_middle_distal": [
            float(args.c_middle),
            float(args.c_distal),
        ],
        "bq_middle_distal": [
            float(args.bq_middle),
            float(args.bq_distal),
        ],
        "integration_dt_s": float(args.dt),
        "dls_lam2": float(args.lam2),
        "max_u_rate_deg_s": float(args.max_u_rate),
        "smoothing_weight": float(args.smoothing),
        "distal_duration_s": args.distal_duration,
        "middle_duration_s": args.middle_duration,
        "home_xy_mm": home_xy.tolist(),
        "home_q4": home_q4.tolist(),
        "raw_model_home_prediction_mm": raw_home_prediction.tolist(),
        "anchor_bias_mm": anchor_bias.tolist(),
        "final_predicted_xy_mm": [
            final["x_hat_mm"],
            final["y_hat_mm"],
        ],
        "final_predicted_error_mm": final["target_error_mm"],
        "final_q4": final_q4.tolist(),
        "final_reduced_u": [
            final["middle_u_deg"],
            final["distal_u_deg"],
        ],
        "maximum_target_error_mm": float(
            max(row["target_error_mm"] for row in log_rows)
        ),
        "minimum_target_error_mm": float(
            min(row["target_error_mm"] for row in log_rows)
        ),
        "maximum_jacobian_condition": float(
            max(row["jacobian_condition"] for row in log_rows)
        ),
        "median_jacobian_condition": float(
            np.median(
                [row["jacobian_condition"] for row in log_rows]
            )
        ),
        "rate_saturation_steps": int(
            sum(int(row["rate_saturated"]) for row in log_rows)
        ),
        "rate_saturation_fraction": float(
            np.mean([row["rate_saturated"] for row in log_rows])
        ),
        "maximum_virtual_task_field_norm": float(
            max(
                math.hypot(row["F_task_x"], row["F_task_y"])
                for row in log_rows
            )
        ),
        "runtime_device": str(device),
        "runtime_measured_steps": int(len(step_times_ms)),
        "runtime_warmup_steps": int(warmup),
        "runtime_step_mean_ms": runtime_mean_ms,
        "runtime_step_median_ms": runtime_median_ms,
        "runtime_step_std_ms": runtime_std_ms,
        "runtime_step_p95_ms": runtime_p95_ms,
        "runtime_step_p99_ms": runtime_p99_ms,
        "runtime_step_max_ms": runtime_max_ms,
        "runtime_mean_frequency_hz": runtime_frequency_hz,
        "runtime_control_period_ms": dt_ms,
        "runtime_deadline_miss_fraction": deadline_miss_fraction,
        "runtime_p95_within_control_period": bool(runtime_p95_ms <= dt_ms),
    }

    if target_q4_reference is not None:
        metrics["measured_target_q4_from_schedule"] = (
            target_q4_reference.tolist()
        )

    save_csv(
        args.output_dir / "reaching_log.csv",
        log_rows,
    )
    with (
        args.output_dir / "metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Downsample the trajectory for hardware replay.
    stride = max(1, int(round(args.command_period / args.dt)))
    replay_rows: List[Dict[str, object]] = []

    selected_indices = list(range(0, len(log_rows), stride))
    if selected_indices[-1] != len(log_rows) - 1:
        selected_indices.append(len(log_rows) - 1)

    for command_id, index in enumerate(selected_indices, start=1):
        row = log_rows[index]

        id1 = int(round(row["ID1"]))
        id4 = int(round(row["ID4"]))
        id5 = int(round(row["ID5"]))
        id6 = int(round(row["ID6"]))

        replay_rows.append(
            {
                "command_id": command_id,
                "time_s": row["time_s"],
                "ID1": id1,
                "ID2": 0,
                "ID3": 0,
                "ID4": id4,
                "ID5": id5,
                "ID6": id6,
                "pred_x_mm": round(row["x_hat_mm"], 3),
                "pred_y_mm": round(row["y_hat_mm"], 3),
                "target_error_mm": round(
                    row["target_error_mm"],
                    3,
                ),
                "arduino_command": (
                    f"G {id1} 0 0 {id4} {id5} {id6}"
                ),
            }
        )

    save_csv(
        args.output_dir / "replay_commands.csv",
        replay_rows,
    )

    print("\nKS-MP command generation completed.")
    print(f"Device: {device}")
    print(
        "HOME measured XY: "
        f"({home_xy[0]:.3f}, {home_xy[1]:.3f}) mm"
    )
    print(
        "Raw network prediction at HOME: "
        f"({raw_home_prediction[0]:.3f}, "
        f"{raw_home_prediction[1]:.3f}) mm"
    )
    print(
        "Applied anchor bias: "
        f"({anchor_bias[0]:+.3f}, "
        f"{anchor_bias[1]:+.3f}) mm"
    )
    print(
        "Target XY: "
        f"({target_xy[0]:.3f}, {target_xy[1]:.3f}) mm"
    )
    if target_q4_reference is not None:
        print(
            "Measured schedule q4 at target sample: "
            f"{target_q4_reference.tolist()}"
        )
    print(
        "Resolved goal u=[middle, distal]: "
        f"{u_goal.round(3).tolist()}"
    )
    print(
        "Learned-model XY at resolved goal: "
        f"({resolved_goal_xy[0]:.3f}, {resolved_goal_xy[1]:.3f}) mm"
    )
    print(
        "Goal resolution discrepancy: "
        f"{resolution_error:.3f} mm"
    )
    print("Reference order: distal first, then middle section")
    print(
        "Unified core parameters: "
        f"dt={args.dt:g} s, "
        f"K=diag({kp_x:g}, {kp_y:g}), "
        f"B=diag({dp_x:g}, {dp_y:g})"
    )
    print(
        "Reduced-coordinate mapping: "
        f"{'J^T' if args.use_jt else 'DLS J^+'}, "
        f"C_u=diag({args.c_middle:g}, {args.c_distal:g}), "
        f"B_Q=diag({args.bq_middle:g}, {args.bq_distal:g})"
    )
    if not args.use_jt:
        print(f"DLS regularisation: lam2={args.lam2:.6g}")
    print(
        "Command constraints: "
        f"max_u_rate={args.max_u_rate:g} deg/s, "
        f"smoothing={args.smoothing:g}"
    )
    print(
        "Generated final q4: "
        f"{final_q4.round(3).tolist()}"
    )
    print(
        "Final predicted XY: "
        f"({final['x_hat_mm']:.3f}, "
        f"{final['y_hat_mm']:.3f}) mm"
    )
    print(
        "Final predicted error: "
        f"{final['target_error_mm']:.3f} mm"
    )
    print(
        "Controller-step runtime "
        f"(excluding first {warmup} warm-up steps):"
    )
    print(
        f"  mean={runtime_mean_ms:.4f} ms, "
        f"median={runtime_median_ms:.4f} ms, "
        f"p95={runtime_p95_ms:.4f} ms, "
        f"p99={runtime_p99_ms:.4f} ms, "
        f"max={runtime_max_ms:.4f} ms"
    )
    print(
        f"  mean frequency={runtime_frequency_hz:.2f} Hz, "
        f"control period={dt_ms:.3f} ms, "
        f"deadline-miss fraction={deadline_miss_fraction:.4f}"
    )
    print(
        "Replay commands: "
        f"{(args.output_dir / 'replay_commands.csv').resolve()}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
