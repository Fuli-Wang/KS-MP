#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-MP controller for a UR10e serial manipulator.

The script provides minimum-jerk, oscillatory, VTGS, and peg-insertion
references; DLS and Jacobian-transpose mappings; diagonal participation and
damping matrices; and runtime, conditioning, and local stability diagnostics.

Units:
- Position: mm
- Joint angles: deg
- Time: s
"""
import argparse
import time
import numpy as np
from dh_fk import UR10e_DH, T_UR10e_TOOL_TO_RG2FT_TCP


# =========================
# Defaults
# =========================
ITERATION_DEFAULT = 1500
DT_DEFAULT        = 0.004
SUBMV_T_DEFAULT   = 6  # s
TRAJ_DEF          = "minjerk"  # vtgs, minjerk, osc, or peg
KP1_DEFAULT       = 100
LAM2_DEFAULT      = 1e-4

# Default joint-space attenuation
BQ_DIAG_DEFAULT   = [0, 0, 0, 0, 0, 0]
# None selects full participation in all joints.
C_DEFAULT         = None

# Shared by the VTGS reference utilities.
RAMP_KONSTANT = DT_DEFAULT
t_dur         = SUBMV_T_DEFAULT


# =========================
# Reference generation
# =========================
def min_jerk_s(t: float, T: float) -> float:
    """Scalar min-jerk phase s(t) in [0, 1] for t in [0, T]."""
    if T <= 0.0:
        return 1.0
    tau = np.clip(t / T, 0.0, 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


def GammaDisc(t_idx: int) -> float:
    """Return the discrete VTGS kernel at one control step."""
    t = t_idx * RAMP_KONSTANT
    if t_dur <= 0.0:
        return 0.0
    tau = np.clip(t / t_dur, 0.0, 1.0)
    # Smooth finite-duration kernel.
    return 30.0 * (tau**2) * ((1.0 - tau)**2)


def Gamma_IntDisc(Gam_arr: np.ndarray, t_idx: int) -> float:
    """Integrate a sampled VTGS kernel up to ``t_idx``."""
    if t_idx <= 0:
        return 0.0
    h = RAMP_KONSTANT
    n = t_idx
    if n % 2 == 1:  # Simpson requires an even number of intervals.
        n -= 1
    if n < 2:
        return Gam_arr[:t_idx+1].sum() * h
    s = Gam_arr[0] + Gam_arr[n]
    s += 4.0 * Gam_arr[1:n:2].sum()
    s += 2.0 * Gam_arr[2:n-1:2].sum()
    return s * (h / 3.0)


# =========================
# KS-MP update and diagnostics
# =========================
def _make_C_and_BQ(args, ndof: int = 6):
    """Build the diagonal participation and attenuation matrices."""
    if args.c_vec is None:
        C_vec = np.ones(ndof, dtype=float)
    else:
        C_vec = np.asarray(args.c_vec, dtype=float)
        if C_vec.size == 1:
            C_vec = np.full(ndof, C_vec[0], dtype=float)
        elif C_vec.size != ndof:
            raise ValueError(f"c_vec must contain 1 or {ndof} values; received {C_vec.size}")
    C_mat = np.diag(C_vec)

    BQ_vec = np.asarray(args.bq_diag, dtype=float)
    if BQ_vec.size == 1:
        BQ_vec = np.full(ndof, BQ_vec[0], dtype=float)
    elif BQ_vec.size != ndof:
        raise ValueError(f"bq_diag must contain 1 or {ndof} values; received {BQ_vec.size}")
    BQ = np.diag(BQ_vec)
    return C_mat, BQ


def _safe_cond_from_svd(svals: np.ndarray, eps: float = 1e-12) -> float:
    if svals.size == 0:
        return np.nan
    smax = float(np.max(svals))
    smin = float(np.min(svals))
    if smin <= eps:
        return np.inf
    return smax / smin


def _spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def _local_stability_diagnostic(J, Kp, Dp, C_mat, BQ, M_map, dt):
    """
    Evaluate the local discrete-time state matrix.

    ``M_map`` is the DLS or scaled Jacobian-transpose differential map.
    """
    task_dim = J.shape[0]
    I_task = np.eye(task_dim)
    H = J @ (np.eye(J.shape[1]) - BQ) @ C_mat @ M_map

    A_top_left = I_task - dt * (H @ Kp) - (H @ Dp)
    A_top_right = H @ Dp
    A_d = np.block([
        [A_top_left, A_top_right],
        [I_task, np.zeros_like(I_task)],
    ])

    H_eigs = np.linalg.eigvals(H)
    H_abs = np.abs(H_eigs)
    return {
        "rho_Ad": _spectral_radius(A_d),
        "H_eig_abs_min": float(np.min(H_abs)),
        "H_eig_abs_max": float(np.max(H_abs)),
        "H_cond_svd": _safe_cond_from_svd(np.linalg.svd(H, compute_uv=False)),
    }


def _jt_scale_from_spectrum(J, C_mat, BQ, args):
    """Spectrally scale the Jacobian-transpose map."""
    P = (np.eye(J.shape[1]) - BQ) @ C_mat
    G = J @ P @ J.T
    G = 0.5 * (G + G.T)  # Enforce numerical symmetry.
    eigvals = np.linalg.eigvalsh(G)
    lam_max = float(np.max(np.abs(eigvals)))
    eps = float(getattr(args, "jt_scale_eps", 1e-12))
    eta = float(args.jt_mu) / (lam_max + eps)
    return eta, lam_max


def core_step(q: np.ndarray, x_ref: np.ndarray, x_prev: np.ndarray, x_ref_prev: np.ndarray, args):
    """
    Execute one KS-MP update.

    The default uses a damped least-squares map. ``--use-jt`` selects a
    fixed or spectrally scaled Jacobian-transpose map.
    """
    t0 = time.perf_counter_ns()

    # End-effector position
    tf0 = time.perf_counter_ns()
    x, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    x = np.asarray(x, dtype=float)
    tf1 = time.perf_counter_ns()

    # Task-space virtual command
    if args.kp1 is not None:
        Kp = np.diag([args.kp1, args.kp1, args.kp1])
    elif (args.kp1_x is not None) or (args.kp1_y is not None) or (args.kp1_z_axis is not None):
        kx = args.kp1_x if args.kp1_x is not None else args.kp1_xy
        ky = args.kp1_y if args.kp1_y is not None else args.kp1_xy
        kz = args.kp1_z_axis if args.kp1_z_axis is not None else args.kp1_z
        Kp = np.diag([kx, ky, kz])
    else:
        kxy = args.kp1_xy
        kz  = args.kp1_z
        Kp = np.diag([kxy, kxy, kz])

    Dp = np.diag([args.dp_x, args.dp_y, args.dp_z])
    e = x_ref - x
    xdot_now = (x - x_prev) / args.dt
    xdot_ref = (x_ref - x_ref_prev) / args.dt
    F_task = Kp @ e + Dp @ (xdot_ref - xdot_now)

    # Participation and attenuation matrices
    C_mat, BQ = _make_C_and_BQ(args, ndof=6)

    # Translational Jacobian
    tj0 = time.perf_counter_ns()
    J = UR10e_DH.jacobian_xyz_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP).astype(float)
    tj1 = time.perf_counter_ns()

    # Differential map
    tm0 = time.perf_counter_ns()
    jt_eta = np.nan
    jt_lam_max = np.nan
    if args.use_jt:
        raw_tau = J.T @ F_task
        if getattr(args, "jt_auto_scale", False):
            jt_eta, jt_lam_max = _jt_scale_from_spectrum(J, C_mat, BQ, args)
        else:
            jt_eta = float(getattr(args, "jt_scale", 1e-3))
        M_map = jt_eta * J.T
        tau = M_map @ F_task
        mapping_name = "JT-auto" if getattr(args, "jt_auto_scale", False) else "JT-fixed"
    else:
        Jpinv = dls_pinv(J, lam2=args.lam2)
        M_map = Jpinv
        tau = M_map @ F_task
        mapping_name = "DLS"
    tm1 = time.perf_counter_ns()

    # Joint update and Euler integration
    qdot_task = C_mat @ tau
    qdot = (np.eye(6) - BQ) @ qdot_task
    q_next = q + qdot * args.dt

    # Numerical diagnostics
    td0 = time.perf_counter_ns()
    svals = np.linalg.svd(J, compute_uv=False)
    diag = _local_stability_diagnostic(J, Kp, Dp, C_mat, BQ, M_map, args.dt)
    td1 = time.perf_counter_ns()
    t1 = time.perf_counter_ns()

    info = {
        "mapping": mapping_name,
        "jt_eta": float(jt_eta) if np.isfinite(jt_eta) else np.nan,
        "jt_lam_max": float(jt_lam_max) if np.isfinite(jt_lam_max) else np.nan,
        "err_norm_mm": float(np.linalg.norm(e)),
        "sigma_min_J": float(np.min(svals)),
        "sigma_max_J": float(np.max(svals)),
        "cond_J": _safe_cond_from_svd(svals),
        "rho_Ad": diag["rho_Ad"],
        "H_eig_abs_min": diag["H_eig_abs_min"],
        "H_eig_abs_max": diag["H_eig_abs_max"],
        "H_cond_svd": diag["H_cond_svd"],
        "step_ms": (t1 - t0) / 1e6,
        "fk_us": (tf1 - tf0) / 1e3,
        "jac_us": (tj1 - tj0) / 1e3,
        "map_us": (tm1 - tm0) / 1e3,
        "diag_us": (td1 - td0) / 1e3,
    }
    return q_next, x, F_task, qdot, info


# =========================
# DLS pseudoinverse
# =========================
def dls_pinv(J: np.ndarray, lam2: float = LAM2_DEFAULT) -> np.ndarray:
    """Damped least-squares pseudo-inverse: Jᵀ (J Jᵀ + λ² I)⁻¹."""
    JT = J.T
    JJt = J @ JT
    lam2I = lam2 * np.eye(JJt.shape[0])
    return JT @ np.linalg.inv(JJt + lam2I)


# =========================
# Controller rollout
# =========================
def run_controller(args, target_xyz: np.ndarray):
    global RAMP_KONSTANT, t_dur
    RAMP_KONSTANT = args.dt
    t_dur         = args.submv_T

    # Initial state
    q = np.asarray(args.q0, dtype=float)
    x0, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    # Previous samples for finite-difference velocity estimates
    x_prev = np.array(x0, dtype=float)
    x_ref_prev = np.array(x0, dtype=float)
    # VTGS integration buffers
    GamX = np.zeros(args.steps, dtype=float)
    GamY = np.zeros(args.steps, dtype=float)
    GamZ = np.zeros(args.steps, dtype=float)

    logs = []
    for t_idx in range(args.steps):
        t = t_idx * args.dt

        # Generate the task-space reference.
        if args.traj == "vtgs":
            Gam = GammaDisc(t_idx)
            GamX[t_idx] = Gam * (target_xyz[0] - x0[0])
            GamY[t_idx] = Gam * (target_xyz[1] - x0[1])
            GamZ[t_idx] = Gam * (target_xyz[2] - x0[2])
            xr = x0[0] + Gamma_IntDisc(GamX, t_idx)
            yr = x0[1] + Gamma_IntDisc(GamY, t_idx)
            zr = x0[2] + Gamma_IntDisc(GamZ, t_idx)
        elif args.traj == "minjerk":
            s = min_jerk_s(t, t_dur)
            xr = x0[0] + s * (target_xyz[0] - x0[0])
            yr = x0[1] + s * (target_xyz[1] - x0[1])
            zr = x0[2] + s * (target_xyz[2] - x0[2])
        elif args.traj == "osc":
            # Circular or drifting oscillation.
            tau = 0.0 if t_dur <= 0.0 else np.clip(t / t_dur, 0.0, 1.0)
            theta = 2.0 * np.pi * args.osc_cycles * tau

            if not args.osc_use_drift:
                # Start at the highest point of the circle.
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
        elif args.traj == "peg":
            depth = abs(float(x0[2]) - float(target_xyz[2]))
            N = max(1, int(round(t_dur / DT_DEFAULT)))
            dz = depth / N
            if dz <= 0.0:
                dz = 1e-6
            z0 = float(x0[2]); zt = float(target_xyz[2])
            direction = 1.0 if (zt - z0) >= 0.0 else -1.0
            z_base = z0 + direction * dz * t_idx
            z_base = float(np.clip(z_base, min(z0, zt), max(z0, zt)))
            # Base XY path during insertion.
            if args.peg_xy_mode == "x0":
                x_base, y_base = float(x0[0]), float(x0[1])
            elif args.peg_xy_mode == "target":
                x_base, y_base = float(target_xyz[0]), float(target_xyz[1])
            else:
                tfrac = float(np.clip(args.peg_xy_tfrac, 1e-6, 1.0))
                t_xy = float(t_dur * tfrac)
                sxy = min_jerk_s(t, t_xy)
                x_base = float(x0[0] + sxy * (target_xyz[0] - x0[0]))
                y_base = float(x0[1] + sxy * (target_xyz[1] - x0[1]))
            theta = 2.0 * np.pi * float(args.peg_osc_freq) * t
            xr = x_base + float(args.peg_osc_amp) * np.cos(theta)
            yr = y_base + float(args.peg_osc_amp) * np.sin(theta)
            zr = z_base
        else:
            raise ValueError(f"Unknown traj type: {args.traj}")

        x_ref = np.array([xr, yr, zr], dtype=float)

        # One KS-MP update.
        q, x, F_task, qdot, info = core_step(q, x_ref, x_prev, x_ref_prev, args)
        # Store samples for the next velocity estimate.
        x_prev = x.copy()
        x_ref_prev = x_ref.copy()

        logs.append([
            t,
            *x.tolist(),
            *x_ref.tolist(),
            *F_task.tolist(),
            *qdot.tolist(),
            *q.tolist(),
            info["err_norm_mm"],
            info["sigma_min_J"], info["sigma_max_J"], info["cond_J"],
            info["rho_Ad"], info["H_eig_abs_min"], info["H_eig_abs_max"], info["H_cond_svd"],
            info["jt_eta"], info["jt_lam_max"],
            info["step_ms"], info["fk_us"], info["jac_us"], info["map_us"], info["diag_us"],
        ])

    # Save compact and full logs.
    logs_arr = np.asarray(logs, dtype=float)
    if logs_arr.size == 0:
        np.savetxt('results.txt', np.zeros((0, 9), dtype=float), fmt='%f')
        np.savetxt(
            'results_head.csv',
            np.zeros((0, 22), dtype=float),
            fmt='%f',
            header='time,x,y,z,x_ref,y_ref,z_ref,Fx,Fy,Fz,qdot1,qdot2,qdot3,qdot4,qdot5,qdot6,q1,q2,q3,q4,q5,q6',
            comments=''
        )
    else:
        # Columns: t, x, x_ref, F_task, qdot, q, diagnostics.
        x_cols = logs_arr[:, 1:4]
        q_cols = logs_arr[:, 16:22]
        basic = np.hstack([q_cols, x_cols])
        np.savetxt('results.txt', basic, fmt='%f')

        header = (
            "time,x,y,z,x_ref,y_ref,z_ref,Fx,Fy,Fz," +
            ",".join([f"qdot{i+1}" for i in range(6)]) + "," +
            ",".join([f"q{i+1}" for i in range(6)]) + "," +
            "err_norm_mm,sigma_min_J,sigma_max_J,cond_J,rho_Ad," +
            "H_eig_abs_min,H_eig_abs_max,H_cond_svd,jt_eta,jt_lam_max," +
            "step_ms,fk_us,jac_us,map_us,diag_us"
        )
        np.savetxt('results_head.csv', logs_arr, fmt='%f', header=header, comments='')

        if getattr(args, "print_summary", True):
            _print_run_summary(logs_arr, args)

    return q, x


# =========================
# Summary printing
# =========================
def _print_run_summary(logs_arr: np.ndarray, args):
    """Print the main metrics for one serial-robot rollout."""
    if logs_arr.size == 0:
        return
    err = logs_arr[:, 22]
    cond = logs_arr[:, 25]
    rho = logs_arr[:, 26]
    jt_eta = logs_arr[:, 30]
    step_ms = logs_arr[:, 32]

    if args.use_jt:
        if getattr(args, "jt_auto_scale", False):
            mapping = f"JT-auto (jt_mu={args.jt_mu:g})"
        else:
            mapping = f"JT-fixed (jt_scale={args.jt_scale:g})"
    else:
        mapping = f"DLS (lam2={args.lam2:g})"

    print("\n=== KS-MP serial run summary ===")
    print(f"  mapping:         {mapping}")
    print(f"  steps/dt/T:      {args.steps} / {args.dt:g} s / {args.submv_T:g} s")
    print(f"  target:          {np.asarray(args.target, dtype=float)}")
    print(f"  mean step time:  {np.mean(step_ms):.6f} ms")
    print(f"  p95  step time:  {np.percentile(step_ms, 95):.6f} ms")
    print(f"  RMS error:       {np.sqrt(np.mean(err**2)):.6g} mm")
    print(f"  mean error:      {np.mean(err):.6g} mm")
    print(f"  final error:     {err[-1]:.6g} mm")
    print(f"  max error:       {np.max(err):.6g} mm")
    print(f"  max cond(J):     {np.nanmax(cond):.6g}")
    print(f"  max rho(A_d):    {np.nanmax(rho):.6g}")
    if args.use_jt:
        finite_eta = jt_eta[np.isfinite(jt_eta)]
        if finite_eta.size:
            print(f"  jt_eta mean:     {np.mean(finite_eta):.6g}")
            print(f"  jt_eta min/max:  {np.min(finite_eta):.6g} / {np.max(finite_eta):.6g}")
    print("  logs:            results_head.csv, results.txt")
    print("===============================\n")


# =========================
# Compatibility wrappers
# =========================
def forward_Kinematics(q):
    xyz, _ = UR10e_DH.fk_xyz_rpy_with_tool(q, T_UR10e_TOOL_TO_RG2FT_TCP)
    return xyz


def VTGS(XT1, YT2, ZT3):
    """Run the controller through the original VTGS entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args([])
    target = np.array([XT1, YT2, ZT3], dtype=float)
    run_controller(args, target)


# =========================
# Command-line interface
# =========================
def _build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=3, default=[-491.73, 181.25, 119.76])
    ap.add_argument("--q0", type=float, nargs=6, default=[180.0, -85.0, -100.0, 0.0, 90.0, 357.58])
    ap.add_argument("--steps", type=int, default=ITERATION_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=SUBMV_T_DEFAULT)

    ap.add_argument("--traj", choices=["vtgs","minjerk","osc","peg"], default=TRAJ_DEF)
    # Oscillation reference
    ap.add_argument("--osc-ax", type=float, default=0.0, help="oscillation amplitude along x [mm]")
    ap.add_argument("--osc-ay", type=float, default=55.0, help="oscillation amplitude along y [mm]")
    ap.add_argument("--osc-cycles", type=float, default=1.0,
                    help="number of oscillation cycles over one primitive duration")
    ap.add_argument("--osc-use-drift", action="store_true",
                    help="add linear drift along x to form S-like trajectories")
    ap.add_argument("--osc-lx", type=float, default=150.0,
                    help="drift length along x [mm] if osc-use-drift is set")

    # Peg-insertion reference
    ap.add_argument("--peg-dz", type=float, default=0.15,
                    help="discrete insertion step along z per control step [mm] (positive value)")
    ap.add_argument("--peg-osc-amp", type=float, default=1.0,
                    help="XY oscillation amplitude during insertion [mm]")
    ap.add_argument("--peg-osc-freq", type=float, default=1.0,
                    help="XY oscillation frequency during insertion [Hz]")
    ap.add_argument("--peg-xy-mode", choices=["x0","target","minjerk"], default="x0",
                    help="base XY during insertion: hold start (x0), hold target, or min-jerk reach to target")
    ap.add_argument("--peg-xy-tfrac", type=float, default=0.2,
                    help="if peg-xy-mode=minjerk: fraction of primitive duration used for XY reaching [0..1]")

    ap.add_argument("--phase1-traj", dest="traj", choices=["vtgs","minjerk","osc","peg"],
                    help=argparse.SUPPRESS)

    ap.add_argument("--kp1", type=float, default=KP1_DEFAULT,
                    help="isotropic task gain (overrides others)")
    ap.add_argument("--kp1-xy", type=float, default=None, help="planar gain for x/y")
    ap.add_argument("--kp1-z",  type=float, default=None, help="vertical gain for z")
    ap.add_argument("--kp1-x", type=float, default=None, help="per-axis gain: x")
    ap.add_argument("--kp1-y", type=float, default=None, help="per-axis gain: y")
    ap.add_argument("--kp1-z-axis", dest="kp1_z_axis", type=float, default=None,
                    help="per-axis gain: z")
    # Task-space damping
    ap.add_argument("--dp-x", type=float, default=0.2,
                    help="task-space damping along x")
    ap.add_argument("--dp-y", type=float, default=0.2,
                    help="task-space damping along y")
    ap.add_argument("--dp-z", type=float, default=0.2,
                    help="task-space damping along z")

    ap.add_argument("--use-jt", action="store_true", default=False,
                    help="use J^T mapping instead of DLS J^+")
    ap.add_argument("--lam2", type=float, default=LAM2_DEFAULT,
                    help="DLS regularization lambda^2")

    # Jacobian-transpose scaling
    ap.add_argument("--jt-scale", type=float, default=1e-3,
                    help="Fixed scaling factor eta for J^T update when --use-jt is enabled.")
    ap.add_argument("--jt-auto-scale", action="store_true", default=True,
                    help="Use spectral normalization for J^T update.")
    ap.add_argument("--jt-mu", type=float, default=2.0,
                    help="Target local task-space gain for auto-scaled J^T update.")
    ap.add_argument("--jt-scale-eps", type=float, default=1e-12,
                    help="Numerical epsilon for J^T auto scaling.")

    # Joint participation and attenuation
    ap.add_argument("--bq-diag", type=float, nargs=6, default=BQ_DIAG_DEFAULT,
                    help="joint damping diagonal entries")
    ap.add_argument("--c-vec", type=float, nargs="+", default=C_DEFAULT,
                    help="KS-MP participation matrix C (1 or 6 values); 0=freeze, 1=full")
    ap.add_argument("--no-print-summary", dest="print_summary", action="store_false",
                    help="Do not print benchmark-style run metrics at the end.")
    ap.set_defaults(print_summary=True)

    return ap


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # Use the drifting oscillation from the reported serial experiment.
    #args.osc_use_drift = True

    # Run the selected primitive.
    q_final, x_final = run_controller(args, np.array(args.target, dtype=float))

    print("[info] goal/primitive execution done. q_final(deg) =", q_final)
    print("[info] x_final(mm) =", x_final)


if __name__ == "__main__":
    main()
