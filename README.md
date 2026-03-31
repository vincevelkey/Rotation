# rotation

Research code for latent-feature models of classical conditioning, including
static Gibbs samplers, dynamic particle-filter experiments, and a
paper-oriented reimplementation of Courville et al. (2003).

## Repository layout

- `static/`: finite-K logistic latent feature model, Gibbs samplers, and
  fixed-structure conditioning experiments.
- `dynamic/`: dynamic IFHMM sampler and latent inhibition / overshadowing
  particle-filter experiments.
- `reimplement/`: self-contained RJMCMC reimplementation of the Courville et al.
  model and experiment notebook.
- `data.py`, `models.py`: compatibility exports that preserve legacy top-level
  imports while the maintained implementations live under `static/`.
- `report/`: bibliography and report assets.

Generated experiment artifacts live under nested `outputs/` directories and are
ignored by default. Local scratch notebooks belong in `trash/`, which is also
ignored.

## Dependencies

The code uses Python with `numpy`, `scipy`, and `polyagamma`. The notebooks and
analysis workflows may also use `matplotlib`, `pandas`, and `scikit-learn`.

## Current status

This repository is organized as an active research workspace. Core model logic
now lives in the package subdirectories, while top-level modules remain as
lightweight import shims for older notebooks and scripts.
