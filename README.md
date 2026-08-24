# Effective Evidence in Correlated LLM Reasoning

## Research Project

This repository contains the research code and experimental infrastructure for studying whether independently sampled large language model (LLM) reasoning trajectories provide independent evidence when used for consensus-based inference.

## Research Question

When multiple LLM reasoning trajectories produce the same answer, does nominal consensus accurately represent the amount of independent evidence supporting that answer?

More specifically, we investigate whether dependence between reasoning trajectories predicts correlated correctness beyond nominal consensus and problem difficulty.

## Core Hypothesis

Repeated LLM reasoning trajectories may exhibit statistical dependence despite being generated independently.

If such dependence is systematically associated with correlated correctness, then the nominal number of reasoning trajectories may overstate the effective amount of evidence supporting a consensus answer.

## Research Direction

The project follows:

Observation
→ Measurement
→ Statistical Validation
→ Theoretical Formulation
→ Algorithmic Development

The initial pilot therefore focuses on measuring trajectory dependence and testing whether it provides predictive information about consensus reliability.

## Current Stage

Stage 1 — Experimental Pilot Infrastructure

The current implementation is intentionally limited to:

- dataset handling;
- repeated trajectory generation;
- deterministic answer evaluation;
- trajectory representation;
- dependence measurement;
- statistical analysis.

The proposed DART adaptive reasoning algorithm will not be implemented until the empirical pilot provides sufficient evidence for its underlying assumptions.

## Repository Structure

```text
configs/          Experiment configurations
data/             Dataset and experiment metadata
src/              Research implementation
experiments/     Reproducible experiment scripts
analysis/         Statistical analysis
results/          Generated results
figures/          Research figures
logs/             Experiment logs
tests/            Automated tests