# rotation

Code for latent-feature models of classical conditioning, including
static Gibbs samplers, dynamic particle-filter experiments, and a
reimplementation of Courville et al. (2003).

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
ignored by default.

## Dependencies

The code uses Python with `numpy`, `scipy`, and `polyagamma`. The notebooks and
analysis workflows may also use `matplotlib`, `pandas`, and `scikit-learn`.

