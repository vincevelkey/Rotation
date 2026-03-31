# Reimplementation

This folder contains a self-contained reimplementation of the sigmoid belief network
and reversible-jump MCMC structure learning setup described in:

- Aaron C. Courville, Nathaniel D. Daw, Geoffrey J. Gordon, and David S. Touretzky,
  "Model Uncertainty in Classical Conditioning," NeurIPS 2003.

## What is implemented

- A binary sigmoid belief network with latent causes and observed stimuli.
- The paper's strong structure prior over:
  - number of latent causes, truncated to `0..5`
  - number of outgoing edges per cause
- Gaussian priors over:
  - active cause-to-stimulus weights
  - latent-cause biases
  - stimulus biases
- RJMCMC moves for:
  - local parameter perturbations
  - edge birth/death
  - latent-cause birth/death
- Exact marginalization over latent causes in the likelihood, which is tractable here
  because the paper's model caps the number of latent causes at five.
- Posterior latent-state sampling for stored draws, so saved samples include
  `C`, `W`, `X`, `b_x`, and `b_y`.

## Layout

- `models.py`: model-facing entry point, matching the top-level repository layout.
- `data.py`: second-order conditioning dataset construction and query helpers.
- `courville_sbn.py`: underlying sampler implementation plus the CLI wrapper.
- `second_order_conditioning_experiment/courville_second_order.ipynb`: notebook that runs the A-X trial sweep and plots the paper-style curves.

## Quick run

From the repository root:

```bash
python3 -m reimplement.courville_sbn --ax-trials 4 --iterations 4000 --burn-in 1000 --thin 10
```

For the full experiment, open the notebook in
`reimplement/second_order_conditioning_experiment/` and run all cells. The notebook
produces the posterior predictive curves for `P(US|A,D)`, `P(US|X,D)`, `P(US|B,D)`,
`P(US|X,B,D)`, and the posterior mean number of latent causes across the A-X sweep.
