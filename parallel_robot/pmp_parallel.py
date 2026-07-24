#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-MP controller for a 6-SPS parallel platform.

The script provides a geometric leg-length model, a numerical differential
map, minimum-jerk and oscillatory references, DLS and Jacobian-transpose
updates, diagonal participation and damping matrices, and CSV logging.

Units:
- Translation and leg length: mm
- Orientation: deg
- Time: s
"""

import argparse
import numpy as np

# =========================
# Default parameters
# =========================
ITERATION_DEFAULT = 1500
DT_DEFAULT        = 0.004
SUBMV_T_DEFAULT   = 6
TRAJ_DEF          = "minjerk"   # "vtgs", "minjerk", or "osc"
OSC_PRIM          = "CBA"
OSC_WAVE          = "sin"   # "sin" or "tri"
OSC_FRAC          = 0.25    # Fraction of one cycle per primitive

KP_DEF_SCALAR     = 100.0
BQ_DIAG_DEFAULT   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LAM2_DEFAULT      = 1e-4

POSE0_DEFAULT     = [0.0, 0.0, 1238.87723, 0.0, 0.0, 0.0]

# None selects full participation in all pose coordinates.
C_DEFAULT         = None

# Shared parameters used by the VTGS helper functions
RAMP_KONSTANT = DT_DEFAULT
t_dur         = SUBMV_T_DEFAULT


# =========================
# Reference and numerical helpers
# =========================
def min_jerk_s(t: float, T: float) -> float:
    tau = np.clip(t / max(T, 1e-9), 0.0, 1.0)
    return (10*tau**3 - 15*tau**4 + 6*tau**5)


def GammaDisc(step_idx: int) -> float:
    t = step_idx * RAMP_KONSTANT
    tau = np.clip(t / max(t_dur, 1e-9), 0.0, 1.0)
    sdot_tau = (30*tau**2 - 60*tau**3 + 30*tau**4)
    return sdot_tau / max(t_dur, 1e-9)


def Gamma_IntDisc(G, n: int) -> float:
    G = np.asarray(G, dtype=float)
    assert 0 <= n < len(G), f"n={n} out of range"
    if n == 0:
        return 0.0
    h = float(RAMP_KONSTANT)
    if n % 2 == 1 and n > 1:
        n_eff = n - 1
    else:
        n_eff = n
    if n_eff < 2:
        return np.sum(G[:n + 1]) * h
    s = G[0] + G[n_eff]
    s += 4.0 * np.sum(G[1:n_eff:2])
    s += 2.0 * np.sum(G[2:n_eff-1:2])
    I_simpson = s * (h / 3.0)
    if n_eff != n:
        I_simpson += 0.5 * h * (G[n_eff] + G[n])
    return I_simpson


def dls_pinv(J: np.ndarray, lam2: float = LAM2_DEFAULT) -> np.ndarray:
    """Damped least-squares pseudo-inverse: Jᵀ (J Jᵀ + λ² I)⁻¹."""
    JT = J.T
    JJt = J @ JT
    lam2I = lam2 * np.eye(JJt.shape[0])
    return JT @ np.linalg.inv(JJt + lam2I)

def tri_wave01(phase: float) -> float:
    """
    Triangular wave in [0,1] as a function of phase (can be any real).
    One period: 0->1->0 over phase in [0,1).
    """
    phase = phase - np.floor(phase)
    if phase < 0.5:
        return 2.0 * phase
    else:
        return 2.0 * (1.0 - phase)

# =========================
# 6-SPS platform geometry
# =========================
def generate_base_and_platform_points_np(base_radius=300.0,
                                         platform_radius=275.0,
                                         z_base=0.0,
                                         z_platform=1238.87723):
    """
    Returns:
        base_points:      (6,3) in base frame
        platform_points:  (6,3) nominal in platform frame (here z=0)
    Angles follow the same pattern as the original code.
    """
    theta_base = np.linspace(0.0, 2.0*np.pi, 6 + 1)[:-1]
    theta_deg = np.array([15, 45, 135, 165, 255, 285], dtype=float)
    theta_platform = np.deg2rad(theta_deg)

    base_points = np.stack([
        base_radius * np.cos(theta_base),
        base_radius * np.sin(theta_base),
        z_base * np.ones(6)
    ], axis=1)

    platform_points = np.stack([
        platform_radius * np.cos(theta_platform),
        platform_radius * np.sin(theta_platform),
        z_platform * np.ones(6)
    ], axis=1)
    # platform frame: set local z=0 (platform lies in its own XY plane)
    platform_points[:, 2] = 0.0
    return base_points, platform_points


# Constant platform geometry
BASE_POINTS, PLATFORM_POINTS = generate_base_and_platform_points_np()
PLATFORM_POINTS_CENTERED = PLATFORM_POINTS - PLATFORM_POINTS.mean(axis=0, keepdims=True)

def rotation_matrix_zyx(roll_deg, pitch_deg, yaw_deg):
    """Rz(yaw) * Ry(pitch) * Rx(roll) in radians."""
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx


def lengths_from_pose(pose):
    """
    pose: [x, y, z, roll, pitch, yaw] in (mm, deg)
    returns: leg lengths L (6,) in mm
    """
    x, y, z, roll, pitch, yaw = pose
    R = rotation_matrix_zyx(roll, pitch, yaw)
    t = np.array([x, y, z], dtype=float)

    P = (R @ PLATFORM_POINTS_CENTERED.T).T + t
    diffs = P - BASE_POINTS
    L = np.linalg.norm(diffs, axis=1)
    return L


def jacobian_lengths_pose(pose, eps=1e-3):
    """
    Numerical Jacobian ∂L/∂pose (6x6) around 'pose' using forward differences.
    pose in (mm, deg), L in mm.
    """
    pose = np.asarray(pose, dtype=float)
    L0 = lengths_from_pose(pose)
    J = np.zeros((6, 6), dtype=float)

    deltas = np.array([eps, eps, eps, 0.1, 0.1, 0.1], dtype=float)

    for i in range(6):
        pose_pert = pose.copy()
        pose_pert[i] += deltas[i]
        Li = lengths_from_pose(pose_pert)
        J[:, i] = (Li - L0) / deltas[i]

    return J


# =========================
# KS-MP control step
# =========================
def core_step(pose, L_ref, L_prev, L_ref_prev, args):
    """
    Advance the platform pose by one KS-MP update.

    Parameters
    ----------
    pose : array_like, shape (6,)
        Current platform pose [x, y, z, roll, pitch, yaw].
    L_ref : array_like, shape (6,)
        Reference leg lengths.
    L_prev, L_ref_prev : array_like, shape (6,)
        Previous measured and reference leg lengths.
    args : argparse.Namespace
        Controller parameters.

    Returns
    -------
    pose_next, L_cur, F_task, pose_rate
    """
    pose = np.asarray(pose, dtype=float)

    # Current geometric leg lengths
    L_cur = lengths_from_pose(pose)

    # Leg-length-space virtual command
    if args.kp_vec is not None and len(args.kp_vec) == 6:
        Kp = np.diag(np.array(args.kp_vec, dtype=float))
    else:
        Kp = np.eye(6) * float(args.kp)

    if args.dp_vec is not None and len(args.dp_vec) == 6:
        Dp = np.diag(np.array(args.dp_vec, dtype=float))
    else:
        Dp = np.eye(6) * float(args.dp)

    eL = np.asarray(L_ref, dtype=float) - L_cur
    # Backward-difference velocity estimates
    Ldot_cur = (L_cur - L_prev) / args.dt
    Ldot_ref = (np.asarray(L_ref, dtype=float) - L_ref_prev) / args.dt
    v_err = Ldot_ref - Ldot_cur

    F_task = Kp @ eL + Dp @ v_err

    # Jacobian: ∂L/∂pose
    J = jacobian_lengths_pose(pose)

    # Differential mapping into pose coordinates
    if args.use_jt:
        tau = J.T @ F_task
    else:
        Jpinv = dls_pinv(J, lam2=args.lam2)
        tau = Jpinv @ F_task

    # Pose-coordinate participation
    if args.c_vec is None:
        C_vec = np.ones(6, dtype=float)
    else:
        C_vec = np.asarray(args.c_vec, dtype=float)
        if C_vec.size == 1:
            C_vec = np.full(6, C_vec[0], dtype=float)
        elif C_vec.size != 6:
            raise ValueError(f"c_vec must contain 1 or 6 values; received {C_vec.size}")
    C_mat = np.diag(C_vec)

    qdot_pose = C_mat @ tau

    # Pose-coordinate damping
    BQ = np.diag(np.array(args.bq_diag, dtype=float))
    qdot_pose = (np.eye(6) - BQ) @ qdot_pose

    # Explicit Euler integration
    pose_next = pose + qdot_pose * args.dt
    return pose_next, L_cur, F_task, qdot_pose


# =========================
# Log formatting
# =========================
def _csv_header():
    return (
        "time,"
        "L1,L2,L3,L4,L5,L6,"
        "Lref1,Lref2,Lref3,Lref4,Lref5,Lref6,"
        "F1,F2,F3,F4,F5,F6,"
        "qdot_pose1,qdot_pose2,qdot_pose3,qdot_pose4,qdot_pose5,qdot_pose6,"
        "x,y,z,roll,pitch,yaw"
    )


def _csv_meta_line(args):
    mapping = "JT" if args.use_jt else "DLS"
    kp_str = "vec" if args.kp_vec is not None else str(args.kp)
    c_str  = "default(1)" if args.c_vec is None else " ".join(map(str, args.c_vec))
    meta = (
        f"# mapping={mapping}, kp={kp_str}, lam2={args.lam2}, "
        f"bq_diag={' '.join(map(str, args.bq_diag))}, "
        f"c_vec={c_str}, "
        f"dt={args.dt}, steps={args.steps}, submv_T={args.submv_T}, "
        f"traj={args.traj}, model=geom"
    )
    return meta


# =========================
# Controller execution
# =========================
def run_controller(args, target_lengths):
    """
    Execute the controller and save the platform trajectory and diagnostics.
    """
    global RAMP_KONSTANT, t_dur
    RAMP_KONSTANT = args.dt
    t_dur         = args.submv_T

    # Initial state
    pose = np.array(args.pose0, dtype=float)
    L0   = lengths_from_pose(pose)

    # VTGS integration buffers
    if args.traj == "vtgs":
        Gam = np.zeros((args.steps, 6), dtype=float)

    logs = []
    L_cur = L0.copy()
    # Previous states for velocity estimation
    L_prev = L0.copy()
    L_ref_prev = L0.copy()

    for t_idx in range(args.steps):
        t = t_idx * args.dt

        # Generate the leg-length reference
        if args.traj == "vtgs":
            L_ref = np.zeros(6, dtype=float)
            for i in range(6):
                Gam[t_idx, i] = GammaDisc(t_idx) * (target_lengths[i] - L0[i])
                s = Gamma_IntDisc(Gam[:, i], t_idx)
                L_ref[i] = L0[i] + s

        elif args.traj == "minjerk":
            s = min_jerk_s(t, t_dur)
            L_ref = L0 + s * (np.asarray(target_lengths, dtype=float) - L0)

        elif args.traj == "osc":
            if not hasattr(run_controller, "_osc_pose_base"):
                run_controller._osc_pose_base = np.asarray(args.pose0, dtype=float).copy()
                run_controller._osc_last_seg  = -1
                run_controller._osc_last_ref  = run_controller._osc_pose_base.copy()

            tau   = 0.0 if t_dur <= 0.0 else np.clip(t / t_dur, 0.0, 1.0)

            if tau == 0.0:
                run_controller._osc_pose_base = np.asarray(args.pose0, dtype=float).copy()
                run_controller._osc_last_seg  = -1
                run_controller._osc_last_ref  = run_controller._osc_pose_base.copy()

            Ax, Ay, Az = args.osc_pos_amp
            Aroll, Apitch, Ayaw = args.osc_ang_amp

            seq = (args.osc_seq.strip().upper() if args.osc_seq else str(args.osc_prim).strip().upper())
            seq = "".join([ch for ch in seq if ch in "ABCD"])
            if len(seq) == 0:
                raise ValueError("Empty oscillation primitive/sequence. Use --osc-prim A|B|C|D or --osc-seq like 'CA'/'ABCD'.")
            nseg = len(seq)

            # Active primitive segment
            seg_idx = min(int(tau * nseg), nseg - 1)

            if seg_idx != run_controller._osc_last_seg and run_controller._osc_last_seg != -1:
                run_controller._osc_pose_base = run_controller._osc_last_ref.copy()
            run_controller._osc_last_seg = seg_idx

            pose_base = run_controller._osc_pose_base
            pose_ref  = pose_base.copy()

            tau0 = seg_idx / nseg
            tau1 = (seg_idx + 1) / nseg
            tau_seg = 0.0 if tau1 <= tau0 else (tau - tau0) / (tau1 - tau0)
            prim = seq[seg_idx]

            phase = args.osc_cycles * args.osc_frac * tau_seg
            # Shared phase variable
            theta = 2.0 * np.pi * phase

            # Primary waveform used by primitives A-C
            if args.osc_wave == "sin":
                s1 = np.sin(theta)
            else:
                s1 = tri_wave01(phase)

            if prim == "A":
                pose_ref[0] = pose_base[0] + Ax     * s1
                pose_ref[4] = pose_base[4] - Apitch * s1

            elif prim == "B":
                pose_ref[1] = pose_base[1] + Ay     * s1
                pose_ref[3] = pose_base[3] + Aroll  * s1

            elif prim == "C":
                pose_ref[2] = pose_base[2] + Az     * s1

            elif prim == "D":
                if args.osc_wave == "sin":
                    # Coupled sinusoidal edge-rolling motion
                    s1 = np.sin(theta)
                    s2 = np.sin(theta) * np.cos(theta)
                else:
                    # Coupled triangular edge-rolling motion
                    w = tri_wave01(phase)
                    s1 = w
                    s2 = 2.0 * w * (1.0 - w)

                pose_ref[0] = pose_base[0] + Ax * s1
                pose_ref[4] = pose_base[4] + Apitch * s1
                pose_ref[1] = pose_base[1] + Ay * s2
                pose_ref[3] = pose_base[3] + Aroll * s2

            run_controller._osc_last_ref = pose_ref.copy()
            L_ref = lengths_from_pose(pose_ref)

        else:
            raise ValueError(f"Unknown traj type: {args.traj}")

        # Apply one KS-MP update
        pose, L_cur, F_task, qdot = core_step(pose, L_ref, L_prev, L_ref_prev, args)
        # Store values for the next velocity estimate
        L_prev = L_cur.copy()
        L_ref_prev = np.asarray(L_ref, dtype=float).copy()

        logs.append([
            t,
            *L_cur.tolist(),
            *L_ref.tolist(),
            *F_task.tolist(),
            *qdot.tolist(),
            *pose.tolist()
        ])

    # Save trajectory logs
    logs_arr = np.asarray(logs, dtype=float)
    if logs_arr.size == 0:
        np.savetxt('results_parallel.txt', np.zeros((0, 12), dtype=float), fmt='%f')
        np.savetxt(
            'results_head_parallel.csv',
            np.zeros((0, 28), dtype=float),
            fmt='%f',
            header=_csv_header(), comments=''
        )
    else:
        L_cols    = logs_arr[:, 1:7]
        pose_cols = logs_arr[:, -6:]
        basic = np.hstack([pose_cols, L_cols])
        np.savetxt('results_parallel.txt', basic, fmt='%f')

        meta   = _csv_meta_line(args)
        header = meta + "\n" + _csv_header()
        np.savetxt('results_head_parallel.csv', logs_arr, fmt='%f', header=header, comments='')

    # Print a model-based summary
    e_final_L2 = np.linalg.norm(np.asarray(target_lengths, dtype=float) -
                                np.asarray(L_cur, dtype=float))
    print("[info] Simulation completed.")
    print(f"[info] Final pose (mm,deg): {pose}")
    print(f"[info] Final leg lengths (geom, mm): {L_cur}")
    if args.traj == "osc":
        print(f"[info] Primitives {args.osc_prim} generated")
    else:
        print(f"[info] Target leg lengths (mm): {np.asarray(target_lengths, dtype=float)}")
        print(f"[info] Final leg-length error norm (geom vs target) = {e_final_L2:.6f} mm")
    print("[info] Files saved: results_parallel.txt, results_head_parallel.csv")
    return pose, L_cur


# =========================
# Compatibility wrappers
# =========================
def VTGS(LT1, LT2, LT3, LT4, LT5, LT6,
         XO1, YO2, ZO3, ChoiceAct, MentalSim, WristGraspPose):
    parser = _build_arg_parser()
    args = parser.parse_args([])  # use defaults when called by legacy
    targets = np.array([LT1, LT2, LT3, LT4, LT5, LT6], dtype=float)
    run_controller(args, targets)


def TargGenSMo():
    target_length = np.zeros(6, dtype=float)
    try:
        with open('target_length.txt', 'r') as f:
            vals = f.read().split()
            for i in range(6):
                target_length[i] = float(vals[i])
    except Exception:
        print("Oops!  Cannot find the target file.")
    VTGS(target_length[0], target_length[1], target_length[2],
         target_length[3], target_length[4], target_length[5],
         0, 0, 0, 0, 0, 0)


# =========================
# Command-line interface
# =========================
def _build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=6,
                    default=[1249.900, 1255.800, 1329.800, 1351.200, 1330.900, 1303.300],
                    help="target leg lengths [mm]")
    ap.add_argument("--pose0", type=float, nargs=6, default=POSE0_DEFAULT,
                    help="initial pose [mm,mm,mm,deg,deg,deg]")

    ap.add_argument("--steps", type=int, default=ITERATION_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--submv-T", dest="submv_T", type=float, default=SUBMV_T_DEFAULT)
    ap.add_argument("--traj", choices=["vtgs", "minjerk", "osc", "legconst"], default=TRAJ_DEF)

    # Pose-space oscillation primitives
    ap.add_argument("--osc-cycles", type=float, default=1.0,
                    help="number of oscillation cycles over one primitive duration")
    ap.add_argument("--osc-prim", default=OSC_PRIM,
                    help="oscillation primitive: A=AP tilt, B=lateral tilt, C=heave, D=edge-rolling")
    ap.add_argument("--osc-seq", type=str, default="",
                help="sequential oscillation primitives, e.g., 'CA' or 'ABCD'. If set, overrides --osc-prim.")
    ap.add_argument("--osc-frac", type=float, default=OSC_FRAC,
                help="fraction of a cycle executed within one primitive duration (e.g., 0.25 for a quarter-cycle)")
    ap.add_argument("--osc-pos-amp", type=float, nargs=3,
                    default=[100.0, 100.0, 220.0],
                    help="max translational offsets [Ax, Ay, Az] in mm for pose oscillation")

    ap.add_argument("--osc-ang-amp", type=float, nargs=3,
                    default=[15.0, 15.0, 5],
                    help="max angular offsets [Aroll, Apitch, Ayaw] in deg for pose oscillation")
    ap.add_argument("--osc-wave", choices=["sin", "tri"],
                    default=OSC_WAVE,
                    help="waveform for pose oscillation: 'sin' (default) or 'tri' (triangular)")
    # Leg-length-space gains
    ap.add_argument("--kp", type=float, default=KP_DEF_SCALAR,
                    help="isotropic gain in length space")
    ap.add_argument("--kp-vec", type=float, nargs=6, dest="kp_vec",
                    help="per-leg gains (overrides --kp if provided)")
    # Leg-length-space damping
    ap.add_argument("--dp", type=float, default=0.2,
                    help="isotropic damping in length space")
    ap.add_argument("--dp-vec", type=float, nargs=6, dest="dp_vec",
                    help="per-leg damping (overrides --dp if provided)")

    ap.add_argument("--use-jt", action="store_true", default=False,
                    help="use J^T mapping instead of DLS J^+")
    ap.add_argument("--lam2", type=float, default=LAM2_DEFAULT,
                    help="DLS regularization lambda^2")

    # Pose-coordinate participation and damping
    ap.add_argument("--bq-diag", type=float, nargs=6, default=BQ_DIAG_DEFAULT,
                    help="pose-side damping diagonal entries")
    ap.add_argument("--c-vec", type=float, nargs=6, default=C_DEFAULT,
                    help="pose-coordinate participation values; 0=disabled, 1=full")

    return ap


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(f"[info] Target lengths (mm): {args.target}")
    print(f"[info] Steps: {args.steps}   dt: {args.dt} s   traj: {args.traj} (T={args.submv_T}s)")
    print(f"[info] Initial pose (mm,deg): {args.pose0}")
    print(f"[info] Mapping: {'J^T' if args.use_jt else 'DLS pinv'}   lam2={args.lam2}")
    print(f"[info] Gains: kp={'vec' if args.kp_vec is not None else args.kp}, "
          f"bq_diag={args.bq_diag}, c_vec={args.c_vec if args.c_vec is not None else 'default(1)'}")

    final_pose, final_L = run_controller(args, np.array(args.target, dtype=float))
    print("[info] done.")


if __name__ == "__main__":
    main()
