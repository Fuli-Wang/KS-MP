# Physical Continuum-Robot Example

This directory provides an optional physical soft-robot case study for the
KS-MP framework.

The example demonstrates how a learned body schema can be constructed from
motor-command and camera-measurement data, differentiated to obtain a local
Jacobian, and used to generate goal-directed commands for a custom
tendon-driven continuum robot.

<p align="center">
  <img src="./continuum_motion_trail.png"
       alt="Custom tendon-driven continuum robot and representative bending motion"
       width="90%">
</p>

<p align="center">
  <em>
  Custom tendon-driven continuum platform and a representative physical
  bending motion.
  </em>
</p>

## Directory contents

```text
physical_continuum_example/
├── README.md
├── hardware.stl
├── continuum_motion_trail.png
├── train_physical_continuum.py
├── pmp_physical_continuum.py
└── example_data/
    ├── sampling_schedule.csv
    └── zed_measurements.csv
```

### `train_physical_continuum.py`

Trains a differentiable body-schema model from paired motor commands and
camera-measured tip positions.

The platform-specific mapping is

```text
q = [ID1, ID4, ID5, ID6]
x = [x_mm, y_mm]
```

where the four inputs are motor commands and the two outputs are the planar
tip position measured using a ZED camera.

The script provides:

- alignment of motor-command and camera-measurement records;
- deterministic train, validation, and test splitting;
- input and output normalisation;
- compact multilayer-perceptron training;
- Cartesian prediction-error metrics;
- model checkpoint export;
- automatic-differentiation Jacobian support.

### `pmp_physical_continuum.py`

Loads the trained body schema and generates a goal-directed KS-MP trajectory.

The script provides:

- learned tip-position prediction;
- learned Jacobian evaluation;
- HOME-position anchoring;
- reduced-coordinate control;
- morphology-compatible two-stage reference generation;
- damped least-squares and Jacobian-transpose mappings;
- diagonal participation and damping terms;
- actuator-rate saturation;
- runtime and Jacobian-conditioning diagnostics;
- downsampled commands for hardware replay.

The script does not communicate directly with the physical robot. It generates
command files that can be transferred to a platform-specific Arduino, motor
controller, or other hardware interface.

### `hardware.stl`

Contains the printable geometry supplied with the physical continuum-robot
example.

Depending on the slicer or CAD software, the STL may be imported as one mesh
or as multiple connected bodies. The file documents the experimental platform
but is not required to run the training or controller scripts.

### `continuum_motion_trail.png`

Shows the printed tendon-driven continuum structure and a representative
physical bending motion.

### `example_data/`

Contains example motor-command and camera-measurement files used to demonstrate
the expected input format.

## Reproducibility scope

This example reproduces the following parts of the physical continuum-robot
workflow:

1. motor-command and ZED-measurement alignment;
2. learned body-schema training;
3. learned-Jacobian evaluation;
4. model-based KS-MP reaching;
5. generation of rate-limited replay commands.

The repository does not provide a universal hardware driver. Motor
communication, tendon tensioning, camera calibration, collision handling,
emergency stopping, and other safety functions remain specific to the
physical platform.

The predicted tracking errors reported by the controller are computed within
the learned body schema. Physical tracking performance must be evaluated
separately using camera or other sensor measurements.

## Requirements

The example requires:

- Python 3;
- NumPy;
- PyTorch.

Install the required packages with:

```bash
pip install numpy torch
```

A CUDA-enabled PyTorch installation can be used when available, but CPU
execution is also supported.

## Data format

The training script aligns the command schedule and ZED measurements using a
shared `sample_id` field.

### Motor-command schedule

The schedule file must contain at least:

```text
sample_id,ID1,ID4,ID5,ID6
```

Example:

```csv
sample_id,ID1,ID4,ID5,ID6
1,0,0,0,0
2,10,10,-10,10
3,20,20,-10,10
```

### ZED tip measurements

The camera-measurement file must contain at least:

```text
sample_id,x_mm,y_mm
```

Example:

```csv
sample_id,x_mm,y_mm
1,52.0,-39.0
2,48.3,-45.7
3,43.1,-51.2
```

The files may be comma-separated, tab-separated, or whitespace-separated,
provided that a valid header row is included.

Rows are matched using `sample_id`. Samples without a corresponding entry in
both files are excluded.

## Train the body schema

Run the training script from this directory:

```bash
python train_physical_continuum.py \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --output-dir checkpoints
```

To reproduce an experiment using only the first 324 aligned samples:

```bash
python train_physical_continuum.py \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --output-dir checkpoints \
  --max-samples 324
```

Important training options include:

| Option | Description |
|---|---|
| `--zed` | ZED measurement file |
| `--schedule` | Motor-command schedule |
| `--output-dir` | Output directory |
| `--max-samples` | Optional limit on aligned samples |
| `--seed` | Random seed |
| `--max-epochs` | Maximum training epochs |
| `--patience` | Early-stopping patience |

The default trained checkpoint is saved as:

```text
checkpoints/physical_continuum_body_schema.pt
```

Other outputs include:

```text
checkpoints/
├── physical_continuum_body_schema.pt
├── metrics.json
├── training_history.csv
├── merged_dataset.csv
└── predictions_all.csv
```

### Training outputs

- `physical_continuum_body_schema.pt`: trained model, normalisation
  parameters, architecture information, and evaluation metadata;
- `metrics.json`: Cartesian prediction-error metrics;
- `training_history.csv`: training and validation losses;
- `merged_dataset.csv`: aligned command and camera data;
- `predictions_all.csv`: measured and predicted tip positions for all samples.

## Actuation manifold

Although the learned model accepts four motor commands,

```text
q4 = [ID1, ID4, ID5, ID6],
```

the collected commands satisfy the platform-specific pair structure

```text
ID1 = ID4 = middle
ID5 = -distal
ID6 = +distal
```

The controller therefore operates in the reduced coordinates

```text
u = [middle, distal].
```

The mapping to the four network inputs is

\[
q_4 = A u,
\]

with

\[
A =
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & -1 \\
0 & 1
\end{bmatrix}.
\]

This prevents the controller from generating motor combinations outside the
actuation manifold represented in the training data.

The reduced Jacobian is

\[
J_u = J_q A,
\]

where

\[
J_q =
\frac{\partial f_\theta(q_4)}
{\partial q_4}.
\]

This reduced-coordinate structure is specific to the supplied platform. It
must be replaced when applying the code to another robot with a different
actuator arrangement.

## Generate a reaching trajectory

After training the body schema, run:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --target-sample 154
```

This uses the measured tip position and motor command associated with sample
154 as the goal.

### Explicit Cartesian target

An explicit planar target can be supplied using:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --zed example_data/zed_measurements.csv \
  --schedule example_data/sampling_schedule.csv \
  --target 10.94 -97.84
```

For an explicit target, the script performs a bounded grid search over the
reduced actuation coordinates and reports the difference between the requested
target and the closest model-resolved goal.

A large goal-resolution discrepancy indicates that the target is outside, or
poorly covered by, the learned workspace.

### Specify the measured HOME position

The learned body schema can be anchored to a measured HOME position:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --home-xy 52 -39 \
  --target-sample 154
```

The anchor bias is

\[
b_h
=
x_{\mathrm{HOME}}^{\mathrm{measured}}
-
f_\theta(q_{\mathrm{HOME}}),
\]

and the corrected prediction is

\[
\hat{x}(q)
=
f_\theta(q)+b_h.
\]

This translation aligns the learned model with the measured HOME point but
does not correct configuration-dependent model errors.

## Two-stage reference

The physical continuum robot uses a morphology-compatible, two-stage
actuation-space reference:

1. the distal section moves first;
2. the middle section moves second while the distal coordinate is held.

The default durations are:

| Stage | Default duration |
|---|---:|
| Distal section | 3 s |
| Middle section | 4 s |
| Final settling period | 1 s |

They can be changed using:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154 \
  --distal-duration 3 \
  --middle-duration 4 \
  --settle-time 1
```

The task-space reference is obtained by passing the staged actuation reference
through the learned body schema.

## Differential mapping

Damped least squares is used by default:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154 \
  --lam2 1e-4
```

The reduced-coordinate update uses

\[
J_{u,\lambda}^{\dagger}
=
J_u^T
\left(
J_uJ_u^T+\lambda^2I
\right)^{-1}.
\]

The Jacobian-transpose implementation can be selected using:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154 \
  --use-jt
```

## Default controller settings

| Parameter | Default value |
|---|---:|
| Mapping | Damped least squares |
| Integration timestep | 0.004 s |
| Cartesian gain | 100 |
| Cartesian damping | 0.2 |
| DLS regularisation | \(10^{-4}\) |
| Middle participation | 1 |
| Distal participation | 1 |
| Reduced-coordinate damping | 0 |
| Maximum coordinate rate | 45 deg/s |
| Command smoothing weight | 1 |
| Replay-command period | 0.20 s |

The Cartesian virtual command is

\[
F_{\mathrm{task}}
=
K_p(x_{\mathrm{ref}}-\hat{x})
+
D_p(\dot{x}_{\mathrm{ref}}-\dot{\hat{x}}).
\]

The reduced-coordinate update is

\[
\dot{u}
=
(I-B_Q)C_uM(F_{\mathrm{task}}),
\]

where \(M\) is either the DLS or Jacobian-transpose differential mapping.

## Participation and damping

The middle- and distal-section participation values can be adjusted using:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154 \
  --c-middle 1 \
  --c-distal 1
```

Reduced-coordinate damping is specified using:

```bash
python pmp_physical_continuum.py \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154 \
  --bq-middle 0 \
  --bq-distal 0
```

## Command constraints

The controller includes platform-specific command constraints.

The maximum coordinate rate is set using:

```bash
--max-u-rate 45
```

An optional first-order command smoother is controlled using:

```bash
--smoothing 1.0
```

A value of `1.0` applies the current saturated command without additional
smoothing. Smaller positive values increase temporal smoothing.

The admissible reduced-coordinate bounds can be configured using:

```bash
--middle-min 0
--middle-max 180
--distal-min 0
--distal-max 180
```

These software bounds do not replace physical limit switches, emergency
stopping, collision checks, or manual supervision.

## Outputs

The controller writes its results to:

```text
physical_reaching_results/
├── reaching_log.csv
├── replay_commands.csv
└── metrics.json
```

### `reaching_log.csv`

Contains the complete model-based rollout, including:

- predicted tip position;
- task-space reference;
- target position;
- reduced coordinates;
- motor commands;
- reference stage;
- task-space virtual command;
- reduced-coordinate rates;
- rate-saturation status;
- Jacobian singular values and condition number;
- controller-step runtime.

### `replay_commands.csv`

Contains downsampled motor commands for platform-specific replay.

The generated command format is:

```text
G ID1 0 0 ID4 ID5 ID6
```

The command period is configured using:

```bash
--command-period 0.20
```

The file is intended as an interface example. Users must verify motor order,
sign conventions, limits, timing, and emergency-stop behaviour before sending
commands to physical hardware.

### `metrics.json`

Contains:

- requested target;
- resolved model goal;
- goal-resolution discrepancy;
- HOME anchor bias;
- final predicted position and error;
- controller parameters;
- Jacobian-conditioning statistics;
- rate-saturation statistics;
- runtime mean, median, p95, p99, and maximum;
- deadline-miss fraction relative to the selected integration period.

All tip-position errors in this file are model-based unless independently
compared with camera measurements.

## CPU execution

To force CPU execution:

```bash
python pmp_physical_continuum.py \
  --cpu \
  --checkpoint checkpoints/physical_continuum_body_schema.pt \
  --target-sample 154
```

## Adapting the example to another robot

The physical example is platform-specific, but the same workflow can be
adapted to another soft or continuum robot.

The following components must be redefined.

### Internal coordinates

Replace

```text
q = [ID1, ID4, ID5, ID6]
```

with the actuator or internal-coordinate vector of the new robot. Possible
coordinates include:

- motor angles;
- tendon displacements;
- chamber pressures;
- cable lengths;
- valve commands;
- estimated shape variables.

### Task variable

Replace

```text
x = [x_mm, y_mm]
```

with the required task variable, such as:

- planar tip position;
- three-dimensional tip position;
- tip pose;
- multiple body points;
- another measurable task-space quantity.

### Training data

Collect paired samples

```text
Q: [N, n]
X: [N, m]
```

and train a differentiable forward model

\[
\hat{x}=f_\theta(q).
\]

The training samples should cover the region in which the controller will be
used. Prediction and Jacobian reliability are not guaranteed outside the
sampled actuation domain.

### Actuation constraints

The supplied platform uses

\[
q=A u.
\]

For another robot, replace `A_REDUCED_TO_Q4` with the appropriate mapping, or
remove the reduced-coordinate layer when all actuator coordinates can be
updated independently.

### Reference generation

Replace the distal-first, middle-second reference with a sequence compatible
with the morphology and actuator arrangement of the new robot.

### Hardware interface

Replace the CSV replay stage with the communication interface required by the
target hardware.

Hardware communication and safety functions should remain separate from the
model-based KS-MP implementation.

## Mechanical file

The included `hardware.stl` provides the printable geometry associated with
this example.

Before printing:

- inspect the mesh in a slicer or CAD program;
- verify that all intended bodies are present;
- check dimensions and units;
- select an appropriate material and print orientation;
- confirm tendon holes and mounting features;
- perform low-load tests before full actuation.

The STL is supplied as an experimental design artifact rather than as a
certified mechanical design.