# rotation

Research code for a finite-K logistic latent feature model with Gibbs sampling via Polya-Gamma augmentation.

## Repository layout

- `models.py`: forward model and Gibbs sampler.
- `data.py`: synthetic data generators and second-order conditioning dataset construction.
- `evals.py`: posterior summary utilities for latent feature usage and weights.
- `post_process.py`: posterior alignment helpers for label switching.
- `second-order_conditioning_experiment/second_order_2.ipynb`: second-order conditioning experiment notebook.
- `validation_results/validation_parametric.ipynb`: parametric validation notebook.
- `validation_results/validation_parametric_eval.ipynb`: repeated parametric evaluation notebook.

## Dependencies

The code uses Python with `numpy`, `scipy`, `matplotlib`, `scikit-learn`, and `polyagamma`.

## Current status

This repository is organized as an active research workspace. Core model logic is in the Python modules, and experiments are currently driven from notebooks.
