# Soft-Robot Implementations

This directory contains the learned-body-schema implementations of KS-MP for
soft and continuum robots.

Unlike the serial- and parallel-robot examples, the soft-robot implementations
do not assume that an analytical kinematic model is available. Instead, a
differentiable forward model is learned from paired actuation and task-space
samples:

\[
\hat{x}=f_\theta(q),
\]

where \(q\) denotes the robot's internal or actuation coordinates and
\(\hat{x}\) denotes the predicted task variable.

The corresponding differential map is obtained through automatic
differentiation:

\[
J_\theta(q)
=
\frac{\partial f_\theta(q)}
{\partial q}.
\]

This learned forward map and Jacobian are then used within the same KS-MP
control structure adopted for the other robot morphologies.

## Directory structure

```text
soft_robot/
├── README.md
├── sorosim/
│   ├── README.md
│   ├── train_softarm.py
│   └── pmp_soft.py
└── physical_continuum_example/
    ├── README.md
    ├── train_physical_continuum.py
    ├── pmp_physical_continuum.py
    ├── hardware.stl
    ├── continuum_motion_trail.png
    └── example_data/
```

## SoRoSim soft arm

The [`sorosim/`](./sorosim/) directory contains the implementation used for
the SoRoSim soft-arm experiments.

It includes:

- training of an 18-input to 3-output learned body schema;
- physical-space prediction-error evaluation;
- training-data coverage diagnostics;
- learned-Jacobian conditioning analysis;
- automatic-differentiation versus finite-difference Jacobian checks;
- KS-MP rollout using the learned forward map and Jacobian;
- trajectory export for subsequent SoRoSim replay.

The Python controller operates within the learned representation. The generated
internal-coordinate trajectory must be replayed in SoRoSim to evaluate the
difference between the learned-model prediction and the simulator-reported tip
position.

See [`sorosim/README.md`](./sorosim/README.md) for dataset requirements,
training commands, controller usage, and output formats.

## Physical continuum-robot example

The
[`physical_continuum_example/`](./physical_continuum_example/)
directory provides an optional physical case study using a custom
tendon-driven continuum robot.

It includes:

- alignment of motor-command and ZED tip-position measurements;
- training of a four-input to two-output learned body schema;
- reduced-coordinate control on the sampled actuation manifold;
- HOME-position anchoring;
- morphology-compatible two-stage reference generation;
- DLS and Jacobian-transpose mappings;
- rate-limited command generation;
- hardware-replay CSV output;
- printable platform geometry and a representative motion image.

This example is platform-specific and is intended to illustrate how the
learned-body-schema workflow can be transferred from simulation to a custom
physical robot.

See
[`physical_continuum_example/README.md`](./physical_continuum_example/README.md)
for the data format, training procedure, controller commands, mechanical file,
and hardware-replay outputs.

## Common workflow

Both implementations follow the same general workflow:

```text
Actuation or internal coordinates
              +
Measured or simulated task variables
              ↓
        Training dataset
              ↓
Differentiable learned body schema
              ↓
 Learned forward map and Jacobian
              ↓
        KS-MP controller
              ↓
Trajectory or hardware commands
              ↓
Independent simulator or sensor evaluation
```

In mathematical form:

\[
q
\longrightarrow
\hat{x}=f_\theta(q),
\]

\[
J_\theta(q)
=
\frac{\partial f_\theta(q)}
{\partial q},
\]

followed by the task-space virtual command

\[
F_{\mathrm{task}}
=
K(x_{\mathrm{ref}}-\hat{x})
+
B(\dot{x}_{\mathrm{ref}}-\dot{\hat{x}}),
\]

and either a damped least-squares or Jacobian-transpose differential mapping.

## Controller-internal and external evaluation

For a learned body schema, controller-internal convergence does not guarantee
exact agreement with the underlying simulator or physical robot.

The following quantities should therefore be distinguished:

1. **Controller-internal tracking deviation**

   The difference between the reference and the learned-model prediction:

   \[
   e_{\mathrm{Pred-Ref}}
   =
   \hat{x}-x_{\mathrm{ref}}.
   \]

2. **External tracking deviation**

   The difference between the reference and the simulator- or sensor-reported
   task variable:

   \[
   e_{\mathrm{External-Ref}}
   =
   x_{\mathrm{external}}-x_{\mathrm{ref}}.
   \]

3. **Model discrepancy**

   The difference between the learned-model prediction and the independent
   simulator or physical measurement:

   \[
   e_{\mathrm{Pred-External}}
   =
   \hat{x}-x_{\mathrm{external}}.
   \]

For the SoRoSim example, the external quantity is the SoRoSim-reported tip
position. For the physical continuum robot, it is the camera-measured tip
position.

## Adapting the workflow to another soft robot

To transfer the implementation to another soft or continuum robot, define:

- an internal-coordinate or actuator vector \(q\);
- a task variable \(x\);
- paired training samples \(Q,X\);
- a differentiable model \(f_\theta(q)\);
- any platform-specific actuation constraints;
- an independent simulator or sensing method for evaluation.

The learned model must provide:

```python
predict(q)
jacobian(q)
```

The prediction and Jacobian are reliable only within the region represented
by the training data. Extrapolation outside the sampled actuation space is not
guaranteed.

## Reproducibility scope

The repository provides:

- learned-body-schema training scripts;
- model evaluation metrics;
- learned-Jacobian diagnostics;
- KS-MP controller implementations;
- simulator-replay or hardware-command outputs;
- an optional physical platform example.

Simulator installation, hardware communication, camera calibration, motor
drivers, collision handling, and emergency-stop functions remain dependent on
the user's platform.