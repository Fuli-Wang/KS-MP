# Parallel Robot Implementation

This directory contains the KS-MP implementation for a geometric 6-SPS
parallel platform.

The platform body schema maps the six-dimensional platform pose to six leg
lengths. Its differential map is evaluated numerically and used within the
same discrete-time KS-MP control structure adopted for the other robot
morphologies in this repository.

## File

### `pmp_parallel.py`

The script provides:

- a geometric model of a 6-SPS Gough–Stewart platform;
- pose-to-leg-length mapping;
- numerical evaluation of the leg-length Jacobian;
- minimum-jerk goal-directed references;
- VTGS references;
- pose-space oscillatory primitives;
- sequential composition of oscillatory primitives;
- damped least-squares and Jacobian-transpose mappings;
- diagonal pose-coordinate participation and damping matrices;
- explicit Euler integration;
- trajectory and controller logging.

## Reproducibility scope

This implementation reproduces the model-based parallel-robot experiments at
the algorithmic level.

The script evolves a platform pose using the geometric 6-SPS model and
generates the corresponding leg-length trajectories. It does not communicate
directly with the physical platform or implement actuator-level feedback.

For the physical demonstrations reported in the accompanying work, the
generated discrete leg-length set-points were transferred to the platform
through a separate hardware playback interface. That hardware-specific
interface is not included in this standalone implementation.

The final leg-length deviations reported by this script are therefore
controller-internal, geometry-based quantities rather than direct physical
measurements.

## Requirements

- Python 3
- NumPy

Install the required package with:

```bash
pip install numpy
```

## Basic usage

Run the examples from this directory:

```bash
cd parallel_robot
```

### Default oscillatory example

The default command executes the predefined oscillatory primitive sequence
using the default platform pose and controller parameters:

```bash
python pmp_parallel.py
```

The current default sequence is:

```text
CBA
```

where the primitives are executed sequentially within one primitive duration.

### Minimum-jerk leg-length reference

Run a goal-directed movement toward a specified set of six leg lengths:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

The target values are expressed in millimetres and ordered as:

```text
L1 L2 L3 L4 L5 L6
```

### Specify the initial platform pose

The platform pose is ordered as:

```text
x y z roll pitch yaw
```

Translations are expressed in millimetres and rotations in degrees:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --pose0 0 0 1238.87723 0 0 0 \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

### VTGS reference

Run the VTGS reference generator in leg-length space:

```bash
python pmp_parallel.py \
  --traj vtgs \
  --target 1249.9 1255.8 1329.8 1351.2 1330.9 1303.3
```

### Single oscillatory primitive

Run a single heave primitive:

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-prim C \
  --osc-pos-amp 100 100 220 \
  --osc-ang-amp 15 15 5
```

### Sequential oscillatory primitives

Run a sequence of primitives:

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

The implementation provides four pose-space primitives.

| Primitive | Pose components |
|---|---|
| `A` | Coupled translation along \(x\) and pitch rotation |
| `B` | Coupled translation along \(y\) and roll rotation |
| `C` | Vertical heave along \(z\) |
| `D` | Coupled edge-rolling motion involving \(x\), \(y\), roll, and pitch |

The translational amplitudes are specified as:

```text
Ax Ay Az
```

using:

```bash
--osc-pos-amp 100 100 220
```

The rotational amplitudes are specified as:

```text
Aroll Apitch Ayaw
```

using:

```bash
--osc-ang-amp 15 15 5
```

The implementation supports sinusoidal and triangular waveforms:

```bash
python pmp_parallel.py \
  --traj osc \
  --osc-seq CBA \
  --osc-wave tri
```

The fraction of one cycle executed by each primitive is controlled through:

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

The numerical differential map is

\[
J_{\lambda}^{\dagger}
=
J^{T}
\left(
JJ^{T}+\lambda^{2}I
\right)^{-1},
\]

where

\[
J = \frac{\partial L}{\partial T}
\]

maps platform-pose variations to leg-length variations.

### Jacobian-transpose mapping

Use the Jacobian-transpose realisation with:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --use-jt
```

The two mappings share the same reference generator, participation matrix,
damping terms, and explicit integration structure.

## Participation matrix

The diagonal participation matrix acts on the six platform-pose coordinates:

```text
x y z roll pitch yaw
```

Full participation is obtained with:

```bash
python pmp_parallel.py \
  --c-vec 1 1 1 1 1 1
```

A selected pose coordinate can be removed from the update by setting its
entry to zero. For example:

```bash
python pmp_parallel.py \
  --traj minjerk \
  --c-vec 1 1 1 1 0 1
```

This example suppresses the pitch-coordinate update.

Such a restricted participation matrix may prevent the platform from reaching
a leg-length target that requires motion in the removed pose direction.
Consequently, a large final deviation should be interpreted as a geometric
compatibility limitation under the selected pose subspace, rather than simply
as numerical tracking noise.

## Gains and damping

### Isotropic leg-length gain

Set a common gain for all six legs:

```bash
python pmp_parallel.py \
  --kp 100
```

### Per-leg gains

Specify six individual gains:

```bash
python pmp_parallel.py \
  --kp-vec 100 100 100 100 100 100
```

When provided, `--kp-vec` overrides `--kp`.

### Isotropic leg-length damping

Set a common damping value:

```bash
python pmp_parallel.py \
  --dp 0.2
```

### Per-leg damping

Specify six individual damping values:

```bash
python pmp_parallel.py \
  --dp-vec 0.2 0.2 0.2 0.2 0.2 0.2
```

When provided, `--dp-vec` overrides `--dp`.

### Pose-coordinate damping

Pose-coordinate damping is specified using six diagonal entries:

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

The platform geometry is defined directly in `pmp_parallel.py` and can be
modified for another 6-SPS mechanism.

## Numerical Jacobian

The leg-length Jacobian is evaluated using forward finite differences:

\[
J
=
\frac{\partial L}
{\partial [x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw}]}.
\]

The default perturbations are:

- \(10^{-3}\) mm for the translational coordinates;
- \(0.1^\circ\) for the rotational coordinates.

The resulting Jacobian combines translational and rotational columns with
different physical units. This convention follows the pose parameterisation
used by the implementation.

## Default settings

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
| DLS regularisation, \(\lambda^2\) | \(10^{-4}\) |
| Participation matrix | \(I_6\) |
| Pose-coordinate damping | \(0_{6\times6}\) |

The default initial pose is:

```text
[0, 0, 1238.87723, 0, 0, 0]
```

The default target leg lengths are:

```text
[1249.9, 1255.8, 1329.8, 1351.2, 1330.9, 1303.3] mm
```

## Coordinate and unit conventions

- Platform pose:
  `[x, y, z, roll, pitch, yaw]`
- Translation:
  millimetres
- Orientation:
  degrees
- Leg lengths:
  millimetres
- Time:
  seconds
- Rotation convention:
  \(R_z(\mathrm{yaw})R_y(\mathrm{pitch})R_x(\mathrm{roll})\)

## Outputs

Each run writes two files to the current working directory.

### `results_head_parallel.csv`

The complete trajectory log contains:

- time;
- current leg lengths;
- reference leg lengths;
- leg-length-space virtual command;
- platform pose-rate update;
- updated platform pose.

The first line also records the principal controller settings, including:

- mapping type;
- gains;
- DLS regularisation;
- damping values;
- participation values;
- timestep;
- number of steps;
- primitive duration;
- reference type.

### `results_parallel.txt`

A compact output containing:

```text
x y z roll pitch yaw L1 L2 L3 L4 L5 L6
```

for every controller step.

## Terminal summary

After execution, the script prints:

- the final platform pose;
- the final geometric leg lengths;
- the target leg lengths for goal-directed runs;
- the final leg-length deviation norm;
- the generated primitive sequence for oscillatory runs;
- the names of the saved result files.

The final deviation is calculated from the geometric model and does not
represent a physical sensor measurement.

## Example reproduction command

The following command explicitly reproduces the default goal-directed
controller configuration:

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

The following command reproduces the default staged oscillatory sequence:

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