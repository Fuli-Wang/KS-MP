# SoRoSim Soft-Arm Implementation

This directory contains the learned-body-schema implementation used for the
SoRoSim soft-arm experiments.

The workflow consists of two stages:

1. train a differentiable body schema from SoRoSim-generated samples;
2. use the learned forward map and its automatic-differentiation Jacobian in
   the KS-MP controller.

The controller generates an internal-coordinate trajectory that can be replayed
in SoRoSim for independent simulator-space evaluation.

## Directory contents

```text
sorosim/
├── README.md
├── train_softarm.py
├── pmp_soft.py
├── data/
    └── softarm_dataset_mm.mat

```

### `train_softarm.py`

Trains and evaluates a differentiable soft-arm body schema

```math
\hat{x}=f_\theta(q),
```

where:

- \(q\) is the vector of internal soft-arm coordinates;
- \(x\) is the Cartesian tip position in millimetres.

The default model uses:

```text
input dimension:  18
output dimension: 3
hidden width:     256
hidden depth:     4
activation:       SiLU
```

The script provides:

- deterministic train, validation, and test splitting;
- training-only normalisation by default;
- physical-space prediction-error metrics;
- nearest-neighbour coverage diagnostics;
- learned-Jacobian condition-number statistics;
- automatic-differentiation versus finite-difference Jacobian checks;
- checkpoint and evaluation-output export.

### `pmp_soft.py`

Loads the trained body schema and runs a KS-MP rollout using:

- learned Cartesian tip prediction;
- an automatic-differentiation Jacobian;
- minimum-jerk, VTGS, or oscillatory references;
- damped least-squares or Jacobian-transpose mapping;
- diagonal participation and damping matrices;
- explicit Euler integration;
- learned-Jacobian conditioning diagnostics;
- controller-step runtime measurements;
- trajectory export for SoRoSim replay.

## Reproducibility scope

The Python controller operates entirely within the learned body-schema
representation.

The controller-internal tip position is

```math
\hat{x}=f_\theta(q),
```

and its Jacobian is

```math
J_\theta(q)
=
\frac{\partial f_\theta(q)}
{\partial q}.
```

Consequently, the tracking deviations reported directly by `pmp_soft.py`
measure convergence within the learned representation.

They do not by themselves measure agreement with the original SoRoSim model.

For simulator-space evaluation, the generated internal-coordinate trajectory
must be replayed in SoRoSim. The resulting simulator-reported tip trajectory
can then be compared with:

- the Cartesian reference;
- the learned-model prediction;
- the final target.

This distinction separates:

1. controller-internal tracking deviation;
2. SoRoSim replay tracking deviation;
3. learned-model prediction versus SoRoSim discrepancy.

## Requirements

The Python scripts require:

- Python 3;
- NumPy;
- SciPy;
- PyTorch.

Install the Python dependencies with:

```bash
pip install numpy scipy torch
```

SoRoSim and MATLAB are required only for generating the original simulation
dataset and replaying the generated internal-coordinate trajectories.

## Dataset format

The training script expects a MATLAB `.mat` file containing:

```text
Q
X
```

with:

```text
Q: [N, n_dof]
X: [N, 3]
```

For the reported SoRoSim soft arm:

```text
Q: [N, 18]
X: [N, 3]
```

Each row of `Q` contains one internal soft-arm configuration, and the
corresponding row of `X` contains the SoRoSim-reported Cartesian tip position
in millimetres.

Example file:

```text
data/softarm_dataset_mm.mat
```

The MATLAB file must contain variables named exactly:

```matlab
Q
X
```

## Train the learned body schema

Run from the `sorosim/` directory:

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_mm.mat \
  --out_dir checkpoints
```

The default training configuration is:

| Parameter | Default |
|---|---:|
| Hidden width | 256 |
| Hidden depth | 4 |
| Batch size | 256 |
| Epochs | 500 |
| Learning rate | \(3\times10^{-4}\) |
| Weight decay | \(10^{-4}\) |
| Train ratio | 0.70 |
| Validation ratio | 0.15 |
| Test ratio | 0.15 |
| Random seed | 42 |

The remaining samples after the train and validation splits form the test set.

### Example with explicit settings

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_mm.mat \
  --out_dir checkpoints \
  --width 256 \
  --depth 4 \
  --batch_size 256 \
  --epochs 500 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --seed 42
```

### Force CPU training

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_mm.mat \
  --out_dir checkpoints \
  --cpu
```

## Training outputs

The training script writes:

```text
checkpoints/
├── softarm_pos_net.pth
├── metrics.csv
├── metrics.json
├── training_history.csv
└── split_indices.npz
```

Optional prediction files are generated when using:

```bash
--save_predictions
```

### `softarm_pos_net.pth`

Contains:

- the trained network state;
- input and output normalisation statistics;
- training arguments;
- the best epoch;
- the best validation loss.

### `metrics.csv` and `metrics.json`

Contain:

- validation and test Cartesian errors;
- per-axis RMSE;
- Euclidean RMSE;
- mean, median, p95, and maximum errors;
- nearest-neighbour coverage statistics;
- Jacobian condition-number statistics;
- automatic-differentiation versus finite-difference Jacobian consistency.

### `split_indices.npz`

Stores the exact train, validation, and test indices for reproducibility.

## Data-efficiency experiments

Different dataset sizes can be evaluated by training separate models from
different `.mat` files.

Example:

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_1000_mm.mat \
  --out_dir checkpoints_1000
```

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_5000_mm.mat \
  --out_dir checkpoints_5000
```

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_10000_mm.mat \
  --out_dir checkpoints_10000
```

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_20000_mm.mat \
  --out_dir checkpoints_20000
```

For a stricter comparison, an external fixed evaluation dataset may be
provided using:

```bash
python train_softarm.py \
  --mat_path data/softarm_dataset_1000_mm.mat \
  --eval_mat_path data/softarm_fixed_test_mm.mat \
  --out_dir checkpoints_1000
```

## Run the KS-MP controller

After training, run:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth
```

The default command uses:

- a minimum-jerk Cartesian reference;
- 1500 controller steps;
- a timestep of 0.004 s;
- a primitive duration of 6 s;
- DLS mapping;
- an isotropic Cartesian gain of 100;
- Cartesian damping of 0.2 per axis;
- DLS regularisation of \(10^{-4}\).

## Goal-directed reaching

Specify a Cartesian target in millimetres:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --traj minjerk \
  --target 790 -11 -364
```

Specify the initial internal coordinates using:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --q0 q1 q2 q3 q4 q5 q6 q7 q8 q9 \
       q10 q11 q12 q13 q14 q15 q16 q17 q18 \
  --target 790 -11 -364
```

The number of `q0` entries must match the input dimension of the trained model.

## Reference types

### Minimum-jerk reference

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --traj minjerk \
  --target 790 -11 -364
```

### VTGS reference

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --traj vtgs \
  --target 790 -11 -364
```

### Oscillatory reference

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --traj osc \
  --osc-ax 50 \
  --osc-ay 50 \
  --osc-cycles 1
```

## Differential mapping

Damped least squares is used by default:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --lam2 1e-4
```

The mapping is

\[
J_{\theta,\lambda}^{\dagger}
=
J_\theta^T
\left(
J_\theta J_\theta^T+\lambda^2I
\right)^{-1}.
\]

Use the Jacobian-transpose implementation with:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --use-jt
```

## Participation and damping

Use full participation in all internal coordinates:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --c-vec 1
```

Provide one value per internal coordinate:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --c-vec 1 1 1 1 1 1 1 1 1 \
          1 1 1 1 1 1 1 1 1
```

Internal-coordinate damping can be specified using:

```bash
python pmp_soft.py \
  --ckpt checkpoints/softarm_pos_net.pth \
  --bq-diag 0
```

A scalar value is broadcast to all model inputs.

## Controller outputs

By default, results are written to:

```text
soft_reaching_results/
```

The output prefix is controlled by:

```bash
--run-name soft_reach
```

A typical output directory contains:

```text
soft_reaching_results/
├── soft_reach_q_traj.csv
├── soft_reach_x_pred_ref.csv
├── soft_reach_results_head.csv
├── soft_reach_results.txt
├── soft_reach_metrics.json
└── soft_reach_metrics.csv
```

### `soft_reach_q_traj.csv`

Contains the generated internal-coordinate trajectory:

```text
q1,q2,...,q18
```

This is the principal file for SoRoSim replay.

### `soft_reach_x_pred_ref.csv`

Contains:

```text
time,x,y,z,x_ref,y_ref,z_ref,ex,ey,ez
```

where `x`, `y`, and `z` are learned-model predictions.

### `soft_reach_results_head.csv`

Contains the complete controller log:

- time;
- predicted tip position;
- Cartesian reference;
- Cartesian virtual command;
- internal-coordinate rate;
- internal-coordinate state.

### `soft_reach_metrics.json`

Contains:

- controller-internal RMS deviation;
- mean, final, p95, and maximum deviation;
- per-axis RMS deviation;
- learned-Jacobian condition statistics;
- controller-step runtime statistics;
- final internal-coordinate state.

## SoRoSim replay

The controller-generated internal-coordinate trajectory should be replayed in
SoRoSim using:

```text
soft_reach_q_traj.csv
```

The SoRoSim replay script should:

1. load each row of the internal-coordinate trajectory;
2. assign the corresponding soft-arm internal coordinates;
3. evaluate or simulate the arm configuration;
4. save the simulator-reported tip position at each step;
5. compare the simulator trajectory with the Cartesian reference and the
   learned-model prediction.

A recommended replay output format is:

```text
time,x_sim,y_sim,z_sim
```

The three principal comparisons are then:

```math
e_{\mathrm{Pred-Ref}}
=
\hat{x}-x_{\mathrm{ref}},
```

```math
e_{\mathrm{Sim-Ref}}
=
x_{\mathrm{sim}}-x_{\mathrm{ref}},
```

and

```math
e_{\mathrm{Pred-Sim}}
=
\hat{x}-x_{\mathrm{sim}}.
```

The Pred–Sim discrepancy measures disagreement between the learned body schema
and the SoRoSim model along the generated trajectory.

## Coordinate alignment

When no explicit initial internal state is supplied, the current implementation
uses the zero internal-coordinate vector and applies a fixed translation to the
learned-model output for the default demonstration.

For a new dataset or robot configuration, the initial state and coordinate
alignment should be explicitly verified. A platform-specific measured or
simulated HOME anchor may be used when appropriate.

## Adapting to another simulated soft robot

The workflow can be transferred to another differentiable or sampled soft-robot
model.

The following elements must be replaced:

1. the internal-coordinate definition \(q\);
2. the task variable \(x\);
3. the sampled dataset \(Q,X\);
4. the model input and output dimensions;
5. the simulator replay interface;
6. platform-specific coordinate alignment and safety constraints.

The learned model must expose:

```python
predict(q)
jacobian(q)
```

with:

```math
\hat{x}=f_\theta(q),
\qquad
J_\theta(q)=\frac{\partial f_\theta(q)}{\partial q}.
```

The reliability of both the prediction and Jacobian depends on the coverage of
the training samples. Extrapolation outside the sampled internal-coordinate
region is not guaranteed.
