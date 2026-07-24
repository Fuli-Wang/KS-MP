# Baseline Benchmarks

This directory contains the repeated algorithmic benchmarks used to compare
KS-MP with conventional reference-tracking and iterative inverse-solving
baselines.

The benchmark scripts import the corresponding robot models from:

```text
serial_robot/
parallel_robot/
```

and should be kept within the repository structure shown below.

## Repository structure

```text
KS-MP/
├── serial_robot/
│   ├── pmp_serial.py
│   └── dh_fk.py
├── parallel_robot/
│   └── pmp_parallel.py
└── baselines/
    ├── README.md
    ├── benchmark_serial.py
    └── benchmark_parallel.py
```

## Benchmark scope

The scripts report algorithmic quantities computed using the same kinematic
body schemas as the main serial- and parallel-robot implementations.

They do not communicate with physical hardware, and the reported tracking
deviations are not physical sensor measurements.

For timing comparisons, run one method per invocation and rotate method order
across repeated trials. The first timing samples are excluded using the
warm-up option.

## Requirements

Install the shared dependencies from the repository root:

```bash
pip install numpy
```

PyTorch is not required for these geometric baseline benchmarks.

## Serial-robot benchmark

`benchmark_serial.py` compares:

| Method key | Description |
|---|---|
| `ksmp-dls` | KS-MP relaxation update with DLS mapping |
| `tracking-dls` | Conventional task-space DLS reference tracking |
| `iterative-ik` | Iterative DLS IK solved at every Cartesian reference point |

All methods use the same UR10e kinematic model, initial configuration,
minimum-jerk Cartesian reference, integration timestep, and evaluation horizon.

### KS-MP

```bash
python baselines/benchmark_serial.py \
  --method ksmp-dls \
  --target -491.73 181.25 119.76 \
  --run-name T1_R1_KSMP
```

### Task-space tracking baseline

```bash
python baselines/benchmark_serial.py \
  --method tracking-dls \
  --target -491.73 181.25 119.76 \
  --run-name T1_R1_TRACK
```

### Iterative IK baseline

```bash
python baselines/benchmark_serial.py \
  --method iterative-ik \
  --target -491.73 181.25 119.76 \
  --run-name T1_R1_IK
```

The default output directory is:

```text
baseline_results/serial/
```

Each invocation saves a method log, a run summary, and updates:

```text
baseline_results/serial/timing_runs.csv
```

## Parallel-robot benchmark

`benchmark_parallel.py` compares:

| Method key | Description |
|---|---|
| `ksmp-dls` | KS-MP relaxation update with configurable participation matrix |
| `leg-pid` | Leg-length PID-style tracking with DLS pose-rate mapping |
| `pose-solve` | Iterative DLS geometric pose solve at every reference point |

All methods use the same 6-SPS geometric body schema, initial pose,
minimum-jerk leg-length reference, integration timestep, and evaluation
horizon.

### KS-MP

```bash
python baselines/benchmark_parallel.py \
  --method ksmp-dls \
  --run-name T1_R1_KSMP
```

### Leg-length tracking baseline

```bash
python baselines/benchmark_parallel.py \
  --method leg-pid \
  --run-name T1_R1_LEG
```

### Iterative pose-solve baseline

```bash
python baselines/benchmark_parallel.py \
  --method pose-solve \
  --run-name T1_R1_SOLVE
```

The default output directory is:

```text
baseline_results/parallel/
```

Each invocation saves:

```text
RUN_log.csv
RUN_step_times.csv
RUN_summary.csv
timing_runs.csv
```

## Timing protocol

The reference is generated outside the timed section.

For KS-MP and the tracking baselines, the timed section includes:

- body-schema evaluation;
- Jacobian evaluation;
- task or tracking command calculation;
- DLS mapping;
- state update.

Diagnostic-only quantities, such as Jacobian condition numbers and local
stability indicators, are evaluated after the timing boundary.

For iterative inverse solving, the full repeated local solve is timed.

The first 10 samples are excluded by default:

```bash
--timing-warmup-steps 10
```

## Interpretation

The baselines have different objectives.

The tracking baselines are explicitly formulated to follow the imposed
minimum-jerk reference and include reference-velocity feedforward where
applicable.

The iterative baselines solve each reference point through repeated local
inverse updates, using the previous solution as the next initial seed.

KS-MP instead applies one regularised differential relaxation update per
controller step. Consequently, tracking deviation alone should not be treated
as a complete measure of equivalence between the methods.

All reported deviations are algorithmic-level, model-based metrics rather than
physical measurements.
