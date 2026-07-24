# Parallel Robot Implementation

This directory contains the KS-MP implementation for a custom 6-SPS
Gough–Stewart parallel platform.

Two body-schema configurations are provided:

1. an **analytic body schema**, which maps platform pose to leg lengths using
   the platform geometry and evaluates the Jacobian numerically;
2. an optional **learned body schema**, supplied as a pretrained checkpoint
   and used in a controlled analytic-versus-learned comparison.

The main KS-MP implementation uses the analytic body schema. The learned model
is included to demonstrate that the controller can retain the same reference,
gains, differential mapping, participation matrix, damping, and integration
structure while replacing only the morphology-specific body schema.

## Directory structure

```text
parallel_robot/
├── README.md
├── pmp_parallel.py
├── compare_body_schemas.py
└── parallel_body_schema.pth
```

## Files

### `pmp_parallel.py`

Main KS-MP implementation for the geometric 6-SPS platform.

The script provides:

- a geometric pose-to-leg-length map;
- numerical evaluation of the leg-length Jacobian;
- minimum-jerk goal-directed references;
- VTGS references;
- pose-space oscillatory primitives;
- sequential composition of oscillatory primitives;
- damped least-squares and Jacobian-transpose mappings;
- diagonal pose-coordinate participation and damping matrices;
- explicit Euler integration;
- trajectory and controller logging.

### `compare_body_schemas.py`

Performs a controlled comparison between:

- analytic geometry with a finite-difference Jacobian;
- a learned pose-to-leg-length map with an
  automatic-differentiation Jacobian.

The two runs use the same:

- initial pose;
- target leg lengths;
- reference trajectory;
- controller gains;
- DLS or Jacobian-transpose mapping;
- participation matrix;
- damping;
- integration timestep;
- evaluation horizon.

Both generated trajectories are additionally evaluated using the analytic
geometry. This separates controller-internal convergence from disagreement
between the learned and analytic body schemas.

### `models/parallel_body_schema.pth`

Pretrained learned body-schema checkpoint used by
`compare_body_schemas.py`.

The checkpoint follows the structure-informed model design described in:

> F. Wang, F. N. Siraj, W. Hutabarat, and A. Tiwari,  
> “Physics-Informed Passive Motion Paradigm for Parallel Robots:
> A High-Precision Motor-Primitives Framework,”  
> *IEEE Robotics and Automation Letters*, vol. 11, no. 2,
> pp. 1874–1881, 2026.  
> [IEEE Xplore](https://ieeexplore.ieee.org/document/11302776) |
> [DOI](https://doi.org/10.1109/LRA.2025.3645663)

The checkpoint is provided to reproduce the body-schema substitution
experiment. The training workflow from the earlier study is not repeated in
this directory.

## Body-schema definitions

The platform pose is represented as

```text
T = [x, y, z, roll, pitch, yaw]
```

and the corresponding actuator state is

```text
L = [L1, L2, L3, L4, L5, L6]
```

### Analytic body schema

The geometric body schema evaluates

```math
L = f_{\mathrm{geom}}(T)
```

from the known base and moving-platform attachment points.

Its differential map is

```math
J_{\mathrm{geom}}
=
\frac{\partial L}{\partial T}
```

In `pmp_parallel.py`, this Jacobian is evaluated using finite differences.

### Learned body schema

The learned model retains the platform pose as the core input. A rigid-body
transformation first maps the local moving-platform attachment points into
the base frame:

```math
P_i^{g}
=
R(\mathrm{roll},\mathrm{pitch},\mathrm{yaw})P_i
+
[x,y,z]^T
```

The transformed attachment points are then processed by a differentiable
neural network to predict the six leg lengths:

```math
\hat{L}
=
f_{\theta}(T)
```

The learned Jacobian is evaluated through automatic differentiation:

```math
J_{\theta}(T)
=
\frac{\partial \hat{L}}{\partial T}
```

Embedding the rigid-body transformation in the forward path preserves an
explicit connection between platform geometry and the learned
pose-to-leg-length mapping.

In the current comparison script, the transformed coordinates of the six
moving-platform points are flattened and passed to a structure-informed neural
network. The hidden width is inferred from the checkpoint when possible and
can be overridden using `--hidden-dim`.

## Reproducibility scope

The scripts reproduce the parallel-robot experiments at the algorithmic level.

`pmp_parallel.py` evolves the platform pose using the geometric 6-SPS body
schema and generates the corresponding leg-length trajectories. It does not
communicate directly with the physical platform or implement actuator-level
feedback.

For the reported physical demonstrations, generated leg-length set-points were
transferred to the platform using a separate hardware playback interface. That
hardware-specific interface is not included here.

The body-schema comparison is also model based. The supplied learned model was
trained using analytically generated pose-to-leg-length relationships.
Therefore, this comparison evaluates substitution of the body-schema
implementation; it should not be interpreted as learning unmodelled physical
effects from sensor data.

## Requirements

The geometric implementation requires:

- Python 3;
- NumPy.

The analytic-versus-learned comparison additionally requires:

- PyTorch.

Install the required packages with:

```bash
pip install numpy torch
```

## Basic usage

Run the examples from this directory:

```bash
cd parallel_robot
```

## Geometric KS-MP implementation

### Default oscillatory example

The default command executes the predefined oscillatory primitive sequence:

```bash
python pmp_parallel.py
```

The current default sequence is:

```text
CBA
```

### Minimum-jerk leg-length reference

Run a goal-directed movement toward six target leg lengths:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

The target order is:

```text
L1 L2 L3 L4 L5 L6
```

### Specify the initial platform pose

Translations are expressed in millimetres and rotations in degrees:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --pose0 0 0 1238.87723 0 0 0 \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

### VTGS reference

```bash
python pmp_parallel.py \
  --traj vtgs \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

### Single oscillatory primitive

Run a heave primitive:

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-prim C \
  --osc-pos-amp 100 100 220 \
  --osc-ang-amp 15 15 5
```

### Sequential oscillatory primitives

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-seq CBA
```

The value supplied through `--osc-seq` overrides `--osc-prim`.

Another example is:

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-seq ABCD \
  --osc-wave sin \
  --osc-frac 0.25
```

## Oscillatory primitives

| Primitive | Pose components |
|---|---|
| `A` | Coupled translation along $x$ and pitch rotation |
| `B` | Coupled translation along $y$ and roll rotation |
| `C` | Vertical heave along $z$ |
| `D` | Coupled edge-rolling motion involving $x$, $y$, roll, and pitch |

Translational amplitudes are ordered as:

```text
Ax Ay Az
```

and specified using:

```bash
--osc-pos-amp 100 100 220
```

Rotational amplitudes are ordered as:

```text
Aroll Apitch Ayaw
```

and specified using:

```bash
--osc-ang-amp 15 15 5
```

Sinusoidal and triangular waveforms are supported:

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-seq CBA \
  --osc-wave tri
```

The fraction of one cycle executed by each active primitive is controlled by:

```bash
--osc-frac 0.25
```

For example, `0.25` corresponds to one quarter of a cycle during the active
primitive segment.

## Differential mapping

### Damped least-squares mapping

Damped least squares is used by default:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --lam2 1e-4
```

The mapping is

```math
J_{\lambda}^{\dagger}
=
J^T
\left(
JJ^T+\lambda^2I
\right)^{-1}
```

### Jacobian-transpose mapping

Use the Jacobian-transpose implementation with:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --use-jt
```

Both mappings use the same reference generator, participation matrix, damping
terms, and explicit integration structure.

## Participation matrix

The diagonal participation matrix acts on:

```text
x y z roll pitch yaw
```

Full participation is obtained with:

```bash
python pmp_parallel.py \
  --c-vec 1 1 1 1 1 1
```

A selected pose coordinate can be removed from the update by setting its entry
to zero:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --c-vec 1 1 1 1 0 1
```

This example suppresses the pitch-coordinate update.

A restricted participation matrix may make a target incompatible with the
remaining pose subspace. A large final deviation under such a setting should
therefore not be interpreted solely as numerical tracking noise.

## Gains and damping

### Isotropic leg-length gain

```bash
python pmp_parallel.py \
  --kp 100
```

### Per-leg gains

```bash
python pmp_parallel.py \
  --kp-vec 100 100 100 100 100 100
```

`--kp-vec` overrides `--kp`.

### Isotropic leg-length damping

```bash
python pmp_parallel.py \
  --dp 0.2
```

### Per-leg damping

```bash
python pmp_parallel.py \
  --dp-vec 0.2 0.2 0.2 0.2 0.2 0.2
```

`--dp-vec` overrides `--dp`.

### Pose-coordinate damping

```bash
python pmp_parallel.py \
  --bq-diag 0 0 0 0 0 0
```

## Platform geometry

The default geometric model uses:

| Parameter | Value |
|---|---:|
| Base radius | 300 mm |
| Moving-platform radius | 275 mm |
| Nominal platform height | 1238.87723 mm |
| Number of legs | 6 |

The base attachment points are uniformly distributed around the base circle.

The moving-platform attachment angles are:

```text
15°, 45°, 135°, 165°, 255°, 285°
```

The geometry is defined directly in the Python scripts and can be modified for
another 6-SPS mechanism.

## Numerical Jacobian

The analytic body schema uses the finite-difference Jacobian

```math
J_{\mathrm{geom}}
=
\frac{\partial L}
{\partial [x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw}]}
```

The default perturbations are:

- $10^{-3}$ mm for translational coordinates;
- $0.1^\circ$ for rotational coordinates.

The Jacobian therefore combines translational and rotational columns with
different physical units.

## Analytic-versus-learned body-schema comparison

Ensure that the checkpoint is located at:

```text
models/parallel_body_schema.pth
```

Then run:

```bash
python compare_body_schemas.py \
  --run-name target_1
```

The default command uses:

- `models/parallel_body_schema.pth`;
- a minimum-jerk leg-length reference;
- 1500 steps;
- a timestep of 0.004 s;
- a duration of 6 s;
- isotropic gain 100;
- isotropic damping 0.2;
- DLS regularisation $10^{-4}$;
- full pose-coordinate participation;
- zero pose-coordinate damping.

### Explicit comparison command

```bash
python compare_body_schemas.py \
  --model models/parallel_body_schema.pth \
  --device auto \
  --traj minjerk \
  --pose0 0 0 1238.87723 0 0 0 \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3 \
  --steps 1500 \
  --dt 0.004 \
  --submv-T 6 \
  --kp 100 \
  --dp 0.2 \
  --lam2 1e-4 \
  --bq-diag 0 0 0 0 0 0 \
  --c-vec 1 1 1 1 1 1 \
  --run-name target_1
```

### Specify a target platform pose

Instead of entering leg lengths directly, a target pose can be converted using
the analytic body schema:

```bash
python compare_body_schemas.py \
  --model models/parallel_body_schema.pth \
  --target-pose 20 -10 1280 2 -3 1 \
  --run-name target_pose_example
```

### Force CPU execution

```bash
python compare_body_schemas.py \
  --model models/parallel_body_schema.pth \
  --device cpu \
  --run-name target_1_cpu
```

### Override the hidden width

The script normally infers the hidden width from the checkpoint. It can be
overridden when required:

```bash
python compare_body_schemas.py \
  --model models/parallel_body_schema.pth \
  --hidden-dim 256 \
  --run-name target_1
```

## Comparison metrics

The comparison separates the following quantities.

### Controller-internal tracking deviation

For the analytic run:

```math
e_{\mathrm{internal}}^{\mathrm{analytic}}
=
L_{\mathrm{ref}}-L_{\mathrm{geom}}
```

For the learned run:

```math
e_{\mathrm{internal}}^{\mathrm{learned}}
=
L_{\mathrm{ref}}-\hat{L}
```

These metrics quantify convergence within the body schema used by each
controller.

### Analytic-geometry replay deviation

Both generated pose trajectories are evaluated through the same analytic
geometry:

```math
e_{\mathrm{geom}}
=
L_{\mathrm{ref}}-L_{\mathrm{geom}}(T)
```

This provides a common model-based evaluation space.

### Forward-model discrepancy

The learned-versus-analytic leg-length discrepancy is

```math
e_{\mathrm{model}}
=
\hat{L}(T)-L_{\mathrm{geom}}(T)
```

### Jacobian discrepancy

At selected diagnostic steps, the script computes

```math
\frac{
\left\|
J_{\theta}-J_{\mathrm{geom}}
\right\|_F
}{
\left\|
J_{\mathrm{geom}}
\right\|_F
}
```

### Runtime

Core controller-step timing excludes the shared post-update diagnostic replay.
The first timing samples can be removed using:

```bash
--timing-warmup 10
```

The learned-versus-analytic Jacobian comparison frequency is controlled by:

```bash
--diagnostic-stride 10
```

## Comparison outputs

For a run named `target_1`, the script writes:

```text
body_schema_comparison_results/
├── target_1_summary.csv
├── target_1_pairwise.csv
├── target_1_analytic_trajectory.csv
├── target_1_learned_trajectory.csv
└── target_1_metadata.json
```

### `target_1_summary.csv`

Contains one summary row for each body schema, including:

- internal tracking RMS;
- analytic-geometry replay RMS;
- final target deviation;
- learned-versus-analytic forward-map discrepancy;
- learned-versus-analytic Jacobian discrepancy;
- Jacobian condition statistics;
- core runtime statistics;
- final platform pose.

### `target_1_pairwise.csv`

Contains direct analytic-versus-learned comparisons, including:

- geometry-replay RMS difference and ratio;
- final geometry-target deviation difference;
- runtime ratio;
- learned internal-to-geometry deviation gap;
- learned forward-model discrepancy;
- learned Jacobian discrepancy.

### Trajectory files

The analytic and learned trajectory files contain per-step:

- reference leg lengths;
- controller-internal leg lengths;
- analytic leg lengths;
- learned leg lengths;
- internal and analytic replay deviations;
- model discrepancy;
- platform pose and pose rate;
- Jacobian conditioning;
- Jacobian discrepancy at diagnostic steps;
- core controller runtime.

### `target_1_metadata.json`

Records:

- checkpoint path;
- device;
- initial pose;
- initial analytic and learned leg lengths;
- target;
- controller parameters;
- output paths;
- interpretation note.

## Default geometric-controller settings

| Parameter | Default value |
|---|---:|
| Differential mapping | Damped least squares |
| Reference type | Oscillatory |
| Default primitive sequence | `CBA` |
| Oscillation waveform | Sinusoidal |
| Cycle fraction | 0.25 |
| Number of steps | 1500 |
| Integration timestep | 0.004 s |
| Primitive duration | 6 s |
| Leg-length gain | 100 |
| Leg-length damping | 0.2 |
| DLS regularisation, $\lambda^2$ | $10^{-4}$ |
| Participation matrix | $I_6$ |
| Pose-coordinate damping | $0_{6\times6}$ |

The default initial pose is:

```text
[0, 0, 1238.87723, 0, 0, 0]
```

The default target leg lengths are:

```text
[1249.9, 1255.8, 1329.8, 1351.2, 1330.9, 1303.3] mm
```

## Coordinate and unit conventions

- Platform pose: `[x, y, z, roll, pitch, yaw]`
- Translation: millimetres
- Orientation: degrees
- Leg lengths: millimetres
- Time: seconds
- Rotation convention:
  $R_z(\mathrm{yaw})R_y(\mathrm{pitch})R_x(\mathrm{roll})$
- Jacobian translation columns: mm/mm
- Jacobian rotation columns: mm/deg

## Geometric-controller outputs

Each `pmp_parallel.py` run writes:

```text
results_head_parallel.csv
results_parallel.txt
```

### `results_head_parallel.csv`

Contains:

- time;
- current leg lengths;
- reference leg lengths;
- leg-length-space virtual command;
- platform pose-rate update;
- updated platform pose;
- principal controller settings in the metadata line.

### `results_parallel.txt`

Contains:

```text
x y z roll pitch yaw L1 L2 L3 L4 L5 L6
```

for each controller step.

The final deviation is calculated from the geometric body schema and does not
represent a physical sensor measurement.

## Reproduction commands

### Goal-directed geometric run

```bash
python pmp_parallel.py \
  --traj minjerk \
  --pose0 0 0 1238.87723 0 0 0 \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3 \
  --steps 1500 \
  --dt 0.004 \
  --submv-T 6 \
  --kp 100 \
  --dp 0.2 \
  --lam2 1e-4 \
  --bq-diag 0 0 0 0 0 0 \
  --c-vec 1 1 1 1 1 1
```

### Staged oscillatory run

```bash
python pmp_parallel.py \
  --traj osc \
  --pose0 0 0 1238.87723 0 0 0 \
  --osc-seq CBA \
  --osc-wave sin \
  --osc-frac 0.25 \
  --osc-pos-amp 100 100 220 \
  --osc-ang-amp 15 15 5 \
  --steps 1500 \
  --dt 0.004 \
  --submv-T 6 \
  --kp 100 \
  --dp 0.2 \
  --lam2 1e-4 \
  --bq-diag 0 0 0 0 0 0 \
  --c-vec 1 1 1 1 1 1
```

### Controlled body-schema comparison

```bash
python compare_body_schemas.py \
  --model models/parallel_body_schema.pth \
  --traj minjerk \
  --pose0 0 0 1238.87723 0 0 0 \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3 \
  --steps 1500 \
  --dt 0.004 \
  --submv-T 6 \
  --kp 100 \
  --dp 0.2 \
  --lam2 1e-4 \
  --bq-diag 0 0 0 0 0 0 \
  --c-vec 1 1 1 1 1 1 \
  --run-name target_1
```

## Checkpoint troubleshooting

### Model file not found

Confirm that the checkpoint is located at:

```text
parallel_robot/models/parallel_body_schema.pth
```

and run the command from `parallel_robot/`.

### Missing or unexpected checkpoint keys

The supplied checkpoint must be compatible with the structure-informed network
implemented in `compare_body_schemas.py`.

A checkpoint from a different model architecture may produce a missing- or
unexpected-key error. Confirm that the correct file is being used, or provide
the correct hidden width through `--hidden-dim`.

### CUDA is unavailable

Use:

```bash
--device cpu
```

### Non-finite controller state

A non-finite state may indicate:

- a target outside the represented workspace;
- poor learned-model coverage;
- an ill-conditioned Jacobian;
- excessive gains;
- insufficient DLS regularisation.

Try a closer target, inspect the reported Jacobian condition number, reduce the
gains, or increase `--lam2`.

## Citation

For the structure-informed learned body schema and its physics-informed
training formulation, cite:

```bibtex
@article{wang2026physicsinformed,
  author  = {Fuli Wang and Fazair Nizar Siraj and Windo Hutabarat and Ashutosh Tiwari},
  title   = {Physics-Informed Passive Motion Paradigm for Parallel Robots:
             A High-Precision Motor-Primitives Framework},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11},
  number  = {2},
  pages   = {1874--1881},
  year    = {2026},
  doi     = {10.1109/LRA.2025.3645663}
}
```

When using the unified KS-MP implementation, please also cite the accompanying
KS-MP article. 
