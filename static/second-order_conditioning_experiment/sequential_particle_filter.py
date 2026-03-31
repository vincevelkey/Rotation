from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import expit, logsumexp

from static.data import generate_data_second_order
from static.models import (
    GibbsSamplerLLFM,
    GibbsSamplerLLFMGeometricMask,
    _geometric_mask_log_prob,
)


AX_TRIAL = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
DEFAULT_CONDS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
DEFAULT_COND_LABELS = {
    str([0, 0, 0]): "none",
    str([1, 0, 0]): "A",
    str([0, 1, 0]): "X",
    str([0, 0, 1]): "B",
    str([0, 1, 1]): "XB",
}


def _bernoulli_log_prob_from_eta(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _bernoulli_log_prob_matrix(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.sum(y[None, :] * eta - np.logaddexp(0.0, eta), axis=1)


def _row_mask_log_prob(row: np.ndarray, rho: float) -> float:
    row = np.asarray(row, dtype=np.int8)
    rho = float(np.clip(rho, 1e-12, 1.0 - 1e-12))
    return float(
        row.sum() * np.log(rho)
        + (row.size - row.sum()) * np.log(1.0 - rho)
    )


def _get_mask_log_prob_fn(model_variant: str):
    if model_variant == "bernoulli":
        return _row_mask_log_prob
    if model_variant == "geometric":
        return _geometric_mask_log_prob
    raise ValueError(f"unknown model_variant={model_variant!r}")


def _get_sampler_class(model_variant: str):
    if model_variant == "bernoulli":
        return GibbsSamplerLLFM
    if model_variant == "geometric":
        return GibbsSamplerLLFMGeometricMask
    raise ValueError(f"unknown model_variant={model_variant!r}")


@dataclass
class Particle:
    W: np.ndarray
    A: np.ndarray
    b: np.ndarray
    Z: np.ndarray

    def copy(self) -> "Particle":
        return Particle(
            W=self.W.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            Z=self.Z.copy(),
        )

    @property
    def T(self) -> int:
        return int(self.Z.shape[0])

    @property
    def K(self) -> int:
        return int(self.W.shape[0])

    @property
    def S(self) -> int:
        return int(self.W.shape[1])

    def occupied_feature_count(self) -> int:
        return int(np.sum(self.Z.sum(axis=0) > 0))

    def effective_weights(self) -> np.ndarray:
        return self.A * self.W


def snapshot_particles(
    particles: list[Particle],
    weights: np.ndarray,
) -> dict[str, object]:
    return {
        "weights": np.asarray(weights, dtype=float).copy(),
        "particles": [particle.copy() for particle in particles],
    }


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_particles = len(weights)
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    return np.searchsorted(cumulative_sum, positions)


def latent_feature_prior_probs(particle: Particle, alpha: float) -> np.ndarray:
    counts = particle.Z.sum(axis=0)
    return np.clip(
        (counts + alpha / particle.K) / (particle.T + 1.0 + alpha / particle.K),
        1e-9,
        1.0 - 1e-9,
    )


def _latent_candidates(
    particle: Particle,
    alpha: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = latent_feature_prior_probs(particle, alpha)
    return rng.binomial(1, probs, size=(n_candidates, particle.K)).astype(np.int8)


def append_trial_via_importance(
    particle: Particle,
    observation: np.ndarray,
    alpha: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> float:
    z_candidates = _latent_candidates(
        particle=particle,
        alpha=alpha,
        n_candidates=n_candidates,
        rng=rng,
    )
    eta_candidates = z_candidates @ particle.effective_weights() + particle.b
    log_likelihood = _bernoulli_log_prob_matrix(observation, eta_candidates)
    log_predictive = float(logsumexp(log_likelihood) - np.log(n_candidates))

    posterior_weights = np.exp(log_likelihood - logsumexp(log_likelihood))
    chosen = int(rng.choice(n_candidates, p=posterior_weights))
    particle.Z = np.vstack([particle.Z, z_candidates[chosen]])
    return log_predictive


def query_probability_from_particle(
    particle: Particle,
    cond_obs: np.ndarray,
    alpha: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> float:
    cond_obs = np.asarray(cond_obs, dtype=float)
    if cond_obs.shape != (particle.S - 1,):
        raise ValueError(f"cond_obs must have shape ({particle.S - 1},), got {cond_obs.shape}")

    z_candidates = _latent_candidates(
        particle=particle,
        alpha=alpha,
        n_candidates=n_candidates,
        rng=rng,
    )
    eta_candidates = z_candidates @ particle.effective_weights() + particle.b
    log_obs = _bernoulli_log_prob_matrix(cond_obs, eta_candidates[:, : particle.S - 1])
    normalized_weights = np.exp(log_obs - logsumexp(log_obs))
    us_prob = expit(eta_candidates[:, -1])
    return float(normalized_weights @ us_prob)


def rejuvenation_sweep(
    particle: Particle,
    data: np.ndarray,
    rho: float,
    alpha: float,
    sigma_w: float,
    sigma_b: float,
    mu_b: np.ndarray,
    model_variant: str = "bernoulli",
) -> None:
    if data.shape[0] != particle.T:
        raise ValueError("particle latent rows and data rows must match during rejuvenation")

    T, S = data.shape
    K = particle.K
    mask_log_prob_fn = _get_mask_log_prob_fn(model_variant)

    eta = particle.Z @ particle.effective_weights() + particle.b
    omega = random_polyagamma(1, eta)

    for s in range(S):
        a = particle.A[:, s]
        active_idx = np.flatnonzero(a)
        if active_idx.size == 0:
            particle.W[:, s] = 0.0
            continue

        omega_s = omega[:, s]
        Z_active = particle.Z[:, active_idx]
        kappa = data[:, s] - 0.5
        z_omega = omega_s[:, None] * Z_active
        precision = Z_active.T @ z_omega + np.eye(active_idx.size) / sigma_w**2
        rhs = Z_active.T @ (kappa - omega_s * particle.b[s])
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        noise = np.linalg.solve(chol.T, np.random.randn(active_idx.size))
        w_new = mean + noise
        particle.W[:, s] = 0.0
        particle.W[active_idx, s] = w_new

    for s in range(S):
        omega_s = omega[:, s]
        kappa = data[:, s] - 0.5
        linear_part = particle.Z @ (particle.A[:, s] * particle.W[:, s])
        var_b = 1.0 / (omega_s.sum() + 1.0 / sigma_b**2)
        mean_b = var_b * (
            np.sum(kappa - omega_s * linear_part) + mu_b[s] / sigma_b**2
        )
        particle.b[s] = mean_b + np.sqrt(var_b) * np.random.randn()

    for s in range(S):
        omega_s = omega[:, s]
        kappa = data[:, s] - 0.5
        eta_s = particle.Z @ (particle.A[:, s] * particle.W[:, s]) + particle.b[s]

        for k in range(K):
            z = particle.Z[:, k]
            old_a = particle.A[k, s]
            old_w = particle.W[k, s]
            current_contrib = z * (old_a * old_w)
            eta_minus = eta_s - current_contrib

            x_omega_x = np.dot(omega_s, z)
            precision = x_omega_x + 1.0 / sigma_w**2
            rhs = np.dot(z, kappa - omega_s * eta_minus)

            row_base = particle.A[k].copy()
            row_base[s] = 0
            row_on = row_base.copy()
            row_on[s] = 1

            logp_on = (
                mask_log_prob_fn(row_on, rho)
                - 0.5 * np.log(precision * sigma_w**2)
                + 0.5 * (rhs**2) / precision
            )
            logp_off = mask_log_prob_fn(row_base, rho)
            prob_on = expit(logp_on - logp_off)
            a_new = np.random.binomial(1, prob_on)

            if a_new == 1:
                var_w = 1.0 / precision
                mean_w = rhs * var_w
                w_new = mean_w + np.sqrt(var_w) * np.random.randn()
            else:
                w_new = 0.0

            particle.A[k, s] = a_new
            particle.W[k, s] = w_new
            eta_s = eta_minus + z * (a_new * w_new)

    w_eff = particle.effective_weights()
    for t in range(T):
        eta_t = particle.Z[t] @ w_eff + particle.b

        for k in range(K):
            z_old = particle.Z[t, k]
            m_k = particle.Z[:, k].sum() - z_old

            log_prior_on = np.log(m_k + alpha / K + 1e-12)
            log_prior_off = np.log(T - m_k + 1e-12)

            contrib = w_eff[k]
            eta_minus = eta_t - contrib if z_old else eta_t

            ll_off = _bernoulli_log_prob_from_eta(data[t], eta_minus)
            eta_on = eta_minus + contrib
            ll_on = _bernoulli_log_prob_from_eta(data[t], eta_on)

            prob_on = expit((ll_on + log_prior_on) - (ll_off + log_prior_off))
            z_new = np.random.binomial(1, prob_on)
            particle.Z[t, k] = z_new

            if z_new != z_old:
                eta_t = eta_minus + z_new * contrib


def initialize_particles(
    n_particles: int,
    *,
    K: int,
    alpha: float,
    rho: float,
    sigma_w: float,
    sigma_b: float,
    mu_b: list[float] | tuple[float, ...],
    n_iter: int,
    burn: int,
    seed: int,
    model_variant: str = "bernoulli",
) -> tuple[list[Particle], GibbsSamplerLLFM]:
    np.random.seed(seed)
    base_data = generate_data_second_order(axnum=0)
    sampler_class = _get_sampler_class(model_variant)
    sampler = sampler_class(
        base_data,
        K=K,
        n_iter=n_iter,
        alpha=alpha,
        rho=rho,
        sigma_w=sigma_w,
        mu_b=list(mu_b),
        sigma_b=sigma_b,
        burn=burn,
        n_subsample=n_particles,
    )
    sampler.run(verbose=False)
    sampler.get_posterior_samples()

    particles = [
        Particle(
            W=sampler.good_samples_W[i].copy(),
            A=sampler.good_samples_A[i].copy(),
            b=sampler.good_samples_b[i].copy(),
            Z=sampler.good_samples_Z[i].copy(),
        )
        for i in range(len(sampler.good_samples_W))
    ]
    return particles, sampler


def weighted_query_probabilities(
    particles: list[Particle],
    weights: np.ndarray,
    conds_list: list[list[int]],
    alpha: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    summaries: dict[str, float] = {}
    for conds in conds_list:
        values = np.array(
            [
                query_probability_from_particle(
                    particle=particle,
                    cond_obs=np.asarray(conds, dtype=float),
                    alpha=alpha,
                    n_candidates=n_candidates,
                    rng=rng,
                )
                for particle in particles
            ],
            dtype=float,
        )
        summaries[str(conds)] = float(weights @ values)
    return summaries


def run_sequential_particle_filter(
    ax_max: int = 60,
    *,
    seed: int = 45,
    n_particles: int = 64,
    K: int = 20,
    alpha: float = 0.1,
    rho: float = 0.1,
    sigma_w: float = 2.0,
    sigma_b: float = 1.0,
    mu_b: list[float] | tuple[float, ...] = (-6.0, -6.0, -6.0, -6.0),
    init_n_iter: int = 2500,
    init_burn: int = 1000,
    latent_update_candidates: int = 256,
    query_candidates: int = 256,
    ess_threshold_fraction: float = 0.5,
    rejuvenation_sweeps_per_resample: int = 1,
    conds_list: list[list[int]] | None = None,
    model_variant: str = "bernoulli",
    snapshot_ax_values: list[int] | tuple[int, ...] | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    particles, base_sampler = initialize_particles(
        n_particles=n_particles,
        K=K,
        alpha=alpha,
        rho=rho,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        mu_b=mu_b,
        n_iter=init_n_iter,
        burn=init_burn,
        seed=seed,
        model_variant=model_variant,
    )

    if conds_list is None:
        conds_list = [list(conds) for conds in DEFAULT_CONDS]

    n_particles = len(particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    mean_occupied = lambda: float(
        weights @ np.array([particle.occupied_feature_count() for particle in particles], dtype=float)
    )

    posterior_probs = {str(conds): [] for conds in conds_list}
    snapshot_ax_values = set(snapshot_ax_values or [])
    snapshots: dict[int, dict[str, object]] = {}
    initial_query = weighted_query_probabilities(
        particles=particles,
        weights=weights,
        conds_list=conds_list,
        alpha=alpha,
        n_candidates=query_candidates,
        rng=rng,
    )
    for conds in conds_list:
        posterior_probs[str(conds)].append(initial_query[str(conds)])
    if 0 in snapshot_ax_values:
        snapshots[0] = snapshot_particles(particles, weights)

    ess_history = [float(n_particles)]
    ess_before_resample_history = [float(n_particles)]
    occupied_history = [mean_occupied()]
    resampled_at: list[int] = []

    for axnum in range(1, ax_max + 1):
        log_weights = np.log(weights + 1e-300)
        for idx, particle in enumerate(particles):
            log_predictive = append_trial_via_importance(
                particle=particle,
                observation=AX_TRIAL,
                alpha=alpha,
                n_candidates=latent_update_candidates,
                rng=rng,
            )
            log_weights[idx] += log_predictive

        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        ess = float(1.0 / np.sum(weights**2))
        ess_before_resample = ess

        if ess < ess_threshold_fraction * n_particles:
            ancestor_idx = systematic_resample(weights, rng)
            particles = [particles[i].copy() for i in ancestor_idx]
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            current_data = generate_data_second_order(axnum=axnum)
            mu_b_arr = np.asarray(mu_b, dtype=float)
            for particle in particles:
                for _ in range(rejuvenation_sweeps_per_resample):
                    rejuvenation_sweep(
                        particle=particle,
                        data=current_data,
                        rho=rho,
                        alpha=alpha,
                        sigma_w=sigma_w,
                        sigma_b=sigma_b,
                        mu_b=mu_b_arr,
                        model_variant=model_variant,
                    )
            resampled_at.append(axnum)
            ess = float(n_particles)

        query_summary = weighted_query_probabilities(
            particles=particles,
            weights=weights,
            conds_list=conds_list,
            alpha=alpha,
            n_candidates=query_candidates,
            rng=rng,
        )
        for conds in conds_list:
            posterior_probs[str(conds)].append(query_summary[str(conds)])

        ess_history.append(ess)
        ess_before_resample_history.append(ess_before_resample)
        occupied_history.append(mean_occupied())
        if axnum in snapshot_ax_values:
            snapshots[axnum] = snapshot_particles(particles, weights)

    return {
        "ax_values": list(range(ax_max + 1)),
        "posterior_probs": posterior_probs,
        "ess_history": ess_history,
        "ess_before_resample_history": ess_before_resample_history,
        "occupied_history": occupied_history,
        "resampled_at": resampled_at,
        "snapshots": snapshots,
        "conds_list": conds_list,
        "cond_labels": DEFAULT_COND_LABELS.copy(),
        "base_sampler": base_sampler,
        "n_particles": n_particles,
        "settings": {
            "seed": seed,
            "K": K,
            "alpha": alpha,
            "rho": rho,
            "sigma_w": sigma_w,
            "sigma_b": sigma_b,
            "mu_b": list(mu_b),
            "init_n_iter": init_n_iter,
            "init_burn": init_burn,
            "latent_update_candidates": latent_update_candidates,
            "query_candidates": query_candidates,
            "ess_threshold_fraction": ess_threshold_fraction,
            "rejuvenation_sweeps_per_resample": rejuvenation_sweeps_per_resample,
            "model_variant": model_variant,
        },
    }
