# KS-MP: Kinematic–Synergy Motor Primitives

A unified and reproducible motor-primitive framework for serial, parallel, and continuum robots.

<p align="center">
  <img src="./assets/ksmp_overview.gif"
       alt="KS-MP demonstrations across serial, parallel, and continuum robots"
       width="100%">
</p>

<p align="center">
  <em>Representative KS-MP demonstrations across three robot morphologies.</em>
</p>

## Overview

KS-MP provides a common discrete-time motor-primitive formulation that can
be instantiated across different robot morphologies through morphology-specific
body schemas and differential mappings.

This repository contains implementations for:

- a UR10e serial manipulator;
- a custom 6-SPS parallel platform;
- simulated and physical continuum robots.

It also includes baseline controllers, trained body-schema models,
experiment configurations, and scripts for reproducing the principal results.

## Citation

When using this implementation, please cite the accompanying KS-MP article.
The full citation will be added following publication.

## Repository structure

```text
KS-MP/
├── assets/
├── serial_robot/
├── parallel_robot/
├── soft_robot/
├── baselines/
├── experiments/
└── results/
