from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from polyagamma import random_polyagamma
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp

from static.data import generate_data_li_ov


DEFAULT_LI_OV_CONDITIONS = [[0, 0], [1, 0], [0, 1], [1, 1]]
DEFAULT_LI_OV_CONDITION_LABELS = {
    str([0, 0]): "none",
    str([1, 0]): "A",
    str([0, 1]): "B",
    str([1, 1]): "AB",
}
LI_OV_SCENARIOS = (
    {"name": "control", "latent_inhibition": False, "overshadowing": False, "label": "No LI / No OV"},
    {"name": "li_only", "latent_inhibition": True, "overshadowing": False, "label": "LI only"},
    {"name": "ov_only", "latent_inhibition": False, "overshadowing": True, "label": "OV only"},
    {"name": "li_ov", "latent_inhibition": True, "overshadowing": True, "label": "LI + OV"},
)


def _bernoulli_log_prob_from_eta(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _bernoulli_log_prob_matrix(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.sum(y[None, :] * eta - np.logaddexp(0.0, eta), axis=1)


def _clip_probability(p: np.ndarray | float, eps: float = 1e-9) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)


def _row_mask_log_prob(row: np.ndarray, rho: float) -> float:
    row = np.asarray(row, dtype=np.int8)
    rho = float(_clip_probability(rho))
    return float(row.sum() * np.log(rho) + (row.size - row.sum()) * np.log(1.0 - rho))


def _transition_prob(prev_state: int, current_state: int, a: float, b_trans: float) -> float:
    if prev_state == 0:
        return a if current_state == 1 else 1.0 - a
    return b_trans if current_state == 1 else 1.0 - b_trans


def _transition_log_prob(prev_state: int, current_state: int, a: float, b_trans: float) -> float:
    return float(np.log(_clip_probability(_transition_prob(prev_state, current_state, a, b_trans))))


def _sample_markov_chain(
    length: int,
    a: float,
    b_trans: float,
    rng: np.random.Generator,
) -> np.ndarray:
    z = np.zeros(length, dtype=np.int8)
    z[0] = rng.binomial(1, _clip_probability(a))
    for t in range(1, length):
        prob_on = b_trans if z[t - 1] == 1 else a
        z[t] = rng.binomial(1, _clip_probability(prob_on))
    return z


def _transition_counts(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = Z.shape[1]
    n00 = np.zeros(K, dtype=int)
    n01 = np.zeros(K, dtype=int)
    n10 = np.zeros(K, dtype=int)
    n11 = np.zeros(K, dtype=int)

    if Z.shape[0] == 0:
        return n00, n01, n10, n11

    # Include the dummy transition from s_0 = 0 into the first observed state.
    n00 += (Z[0] == 0).astype(int)
    n01 += (Z[0] == 1).astype(int)

    if Z.shape[0] > 1:
        prev = Z[:-1]
        curr = Z[1:]
        n00 += np.sum((prev == 0) & (curr == 0), axis=0)
        n01 += np.sum((prev == 0) & (curr == 1), axis=0)
        n10 += np.sum((prev == 1) & (curr == 0), axis=0)
        n11 += np.sum((prev == 1) & (curr == 1), axis=0)
    return n00.astype(int), n01.astype(int), n10.astype(int), n11.astype(int)


def _safe_generate_li_ov_data(
    trials: int = 50,
    latent_inhibition: bool = True,
    overshadowing: bool = True,
) -> np.ndarray:
    try:
        return np.asarray(
            generate_data_li_ov(
                trials=trials,
                latent_inhibition=latent_inhibition,
                overshadowing=overshadowing,
            ),
            dtype=float,
        )
    except TypeError:
        if trials <= 0:
            trials = 50
        total = 50 + trials + (50 if latent_inhibition else 0)
        data = np.zeros((total, 3), dtype=float)
        next_row = 50
        if latent_inhibition:
            data[next_row:next_row + 50, :] = [1.0, 0.0, 0.0]
            next_row += 50
        if overshadowing:
            data[next_row:, :] = [1.0, 1.0, 1.0]
        else:
            data[next_row:, :] = [1.0, 0.0, 1.0]
        return data


@dataclass
class DynamicState:
    W: np.ndarray
    A: np.ndarray
    b: np.ndarray
    Z: np.ndarray
    a: np.ndarray
    b_trans: np.ndarray

    def copy(self) -> "DynamicState":
        return DynamicState(
            W=self.W.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            Z=self.Z.copy(),
            a=self.a.copy(),
            b_trans=self.b_trans.copy(),
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

    def effective_weights(self) -> np.ndarray:
        return self.A * self.W

    def linear_predictor(self) -> np.ndarray:
        return self.Z @ self.effective_weights() + self.b

    def sample_observations(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        probs = expit(self.linear_predictor())
        return rng.binomial(1, probs)

    def occupied_feature_count(self, threshold: int = 1) -> int:
        return int(np.sum(self.Z.sum(axis=0) >= threshold))

    def transition_counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _transition_counts(self.Z)


class DynamicIFHMMModel:
    """
    Truncated approximation to the infinite factorial hidden Markov model.

    Each latent feature is a binary Markov chain with its own initial activation
    probability and transition probabilities. Bernoulli-logistic emissions are
    produced by the sum of all active feature contributions.
    """

    def __init__(
        self,
        T: int,
        S: int,
        K: int,
        rho: float = 0.2,
        alpha: float = 1.0,
        sigma_w: float = 2.0,
        sigma_b: float = 1.0,
        mu_b: list[float] | tuple[float, ...] | np.ndarray | None = None,
        gamma: float = 6.0,
        delta: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.T = T
        self.S = S
        self.K = K
        self.rho = rho
        self.alpha = alpha
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.mu_b = np.full(S, -3.0, dtype=float) if mu_b is None else np.asarray(mu_b, dtype=float)
        self.gamma = gamma
        self.delta = delta
        self.rng = np.random.default_rng() if rng is None else rng
        self.state = self._sample_from_prior()

    def _sample_from_prior(self) -> DynamicState:
        a = self.rng.beta(self.alpha / self.K, 1.0, size=self.K)
        b_trans = self.rng.beta(self.gamma, self.delta, size=self.K)
        A = self.rng.binomial(1, _clip_probability(self.rho), size=(self.K, self.S)).astype(np.int8)
        W = self.rng.normal(0.0, self.sigma_w, size=(self.K, self.S))
        W *= A
        b = self.rng.normal(self.mu_b, self.sigma_b, size=self.S)
        Z = np.column_stack(
            [_sample_markov_chain(self.T, a[k], b_trans[k], self.rng) for k in range(self.K)]
        ).astype(np.int8)
        return DynamicState(W=W, A=A, b=b, Z=Z, a=a, b_trans=b_trans)

    def sample_observations(self) -> np.ndarray:
        return self.state.sample_observations(self.rng)


def _sample_weights(
    state: DynamicState,
    data: np.ndarray,
    sigma_w: float,
    omega: np.ndarray | None = None,
) -> None:
    if omega is None:
        omega = random_polyagamma(1, state.linear_predictor())

    for s in range(state.S):
        active_idx = np.flatnonzero(state.A[:, s])
        if active_idx.size == 0:
            state.W[:, s] = 0.0
            continue

        omega_s = omega[:, s]
        Z_active = state.Z[:, active_idx]
        kappa = data[:, s] - 0.5
        z_omega = omega_s[:, None] * Z_active
        precision = Z_active.T @ z_omega + np.eye(active_idx.size) / sigma_w**2
        rhs = Z_active.T @ (kappa - omega_s * state.b[s])
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        noise = np.linalg.solve(chol.T, np.random.randn(active_idx.size))
        state.W[:, s] = 0.0
        state.W[active_idx, s] = mean + noise


def _sample_biases(
    state: DynamicState,
    data: np.ndarray,
    sigma_b: float,
    mu_b: np.ndarray,
    omega: np.ndarray | None = None,
) -> None:
    if omega is None:
        omega = random_polyagamma(1, state.linear_predictor())

    for s in range(state.S):
        omega_s = omega[:, s]
        kappa = data[:, s] - 0.5
        linear_part = state.Z @ state.effective_weights()[:, s]
        var_b = 1.0 / (omega_s.sum() + 1.0 / sigma_b**2)
        mean_b = var_b * (np.sum(kappa - omega_s * linear_part) + mu_b[s] / sigma_b**2)
        state.b[s] = mean_b + np.sqrt(var_b) * np.random.randn()


def _sample_masks(
    state: DynamicState,
    data: np.ndarray,
    rho: float,
    sigma_w: float,
    omega: np.ndarray | None = None,
) -> None:
    if omega is None:
        omega = random_polyagamma(1, state.linear_predictor())

    for s in range(state.S):
        omega_s = omega[:, s]
        kappa = data[:, s] - 0.5
        eta_s = state.Z @ (state.A[:, s] * state.W[:, s]) + state.b[s]

        for k in range(state.K):
            z = state.Z[:, k]
            old_a = state.A[k, s]
            old_w = state.W[k, s]
            current_contrib = z * (old_a * old_w)
            eta_minus = eta_s - current_contrib

            x_omega_x = np.dot(omega_s, z)
            precision = x_omega_x + 1.0 / sigma_w**2
            rhs = np.dot(z, kappa - omega_s * eta_minus)

            row_base = state.A[k].copy()
            row_base[s] = 0
            row_on = row_base.copy()
            row_on[s] = 1

            logp_on = _row_mask_log_prob(row_on, rho) - 0.5 * np.log(precision * sigma_w**2) + 0.5 * (rhs**2) / precision
            logp_off = _row_mask_log_prob(row_base, rho)
            prob_on = expit(logp_on - logp_off)
            a_new = np.random.binomial(1, prob_on)

            if a_new == 1:
                var_w = 1.0 / precision
                mean_w = rhs * var_w
                w_new = mean_w + np.sqrt(var_w) * np.random.randn()
            else:
                w_new = 0.0

            state.A[k, s] = a_new
            state.W[k, s] = w_new
            eta_s = eta_minus + z * (a_new * w_new)


def _sample_latent_sequences(state: DynamicState, data: np.ndarray) -> None:
    w_eff = state.effective_weights()
    eta = state.Z @ w_eff + state.b

    for k in range(state.K):
        contrib = w_eff[k]
        for t in range(state.T):
            z_old = state.Z[t, k]
            eta_minus = eta[t] - z_old * contrib

            ll0 = _bernoulli_log_prob_from_eta(data[t], eta_minus)
            ll1 = _bernoulli_log_prob_from_eta(data[t], eta_minus + contrib)

            if t == 0:
                log_prior0 = _transition_log_prob(0, 0, state.a[k], state.b_trans[k])
                log_prior1 = _transition_log_prob(0, 1, state.a[k], state.b_trans[k])
            else:
                prev_state = int(state.Z[t - 1, k])
                log_prior0 = _transition_log_prob(prev_state, 0, state.a[k], state.b_trans[k])
                log_prior1 = _transition_log_prob(prev_state, 1, state.a[k], state.b_trans[k])

            if t < state.T - 1:
                next_state = int(state.Z[t + 1, k])
                log_prior0 += _transition_log_prob(0, next_state, state.a[k], state.b_trans[k])
                log_prior1 += _transition_log_prob(1, next_state, state.a[k], state.b_trans[k])

            prob_on = expit((ll1 + log_prior1) - (ll0 + log_prior0))
            z_new = np.random.binomial(1, prob_on)
            state.Z[t, k] = z_new

            if z_new != z_old:
                eta[t] = eta_minus + z_new * contrib


def _sample_transition_parameters(
    state: DynamicState,
    alpha: float,
    gamma: float,
    delta: float,
) -> None:
    n00, n01, n10, n11 = state.transition_counts()
    state.a = np.random.beta(alpha / state.K + n01, 1.0 + n00)
    state.b_trans = np.random.beta(gamma + n11, delta + n10)


class DynamicIFHMMSampler:
    def __init__(
        self,
        Data: np.ndarray,
        K: int = 20,
        rho: float = 0.2,
        alpha: float = 1.0,
        sigma_w: float = 2.0,
        sigma_b: float = 1.0,
        mu_b: list[float] | tuple[float, ...] | np.ndarray | None = None,
        n_iter: int = 1000,
        burn: int = 200,
        n_subsample: int | None = None,
        gamma: float = 6.0,
        delta: float = 1.0,
    ):
        self.Data = np.asarray(Data, dtype=float)
        self.T, self.S = self.Data.shape
        self.K = K
        self.rho = rho
        self.alpha = alpha
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.mu_b = np.full(self.S, -3.0, dtype=float) if mu_b is None else np.asarray(mu_b, dtype=float)
        self.n_iter = n_iter
        self.burn = burn
        self.n_subsample = n_subsample
        self.gamma = gamma
        self.delta = delta

        model = DynamicIFHMMModel(
            T=self.T,
            S=self.S,
            K=self.K,
            rho=self.rho,
            alpha=self.alpha,
            sigma_w=self.sigma_w,
            sigma_b=self.sigma_b,
            mu_b=self.mu_b,
            gamma=self.gamma,
            delta=self.delta,
        )
        self.state = model.state

        self.samples_W = np.empty((self.n_iter, self.K, self.S))
        self.samples_A = np.empty((self.n_iter, self.K, self.S), dtype=np.int8)
        self.samples_b = np.empty((self.n_iter, self.S))
        self.samples_Z = np.empty((self.n_iter, self.T, self.K), dtype=np.int8)
        self.samples_a = np.empty((self.n_iter, self.K))
        self.samples_b_trans = np.empty((self.n_iter, self.K))

        sample_count = 0 if self.n_subsample is None else self.n_subsample
        self.good_samples_W = np.empty((sample_count, self.K, self.S))
        self.good_samples_A = np.empty((sample_count, self.K, self.S), dtype=np.int8)
        self.good_samples_b = np.empty((sample_count, self.S))
        self.good_samples_Z = np.empty((sample_count, self.T, self.K), dtype=np.int8)
        self.good_samples_a = np.empty((sample_count, self.K))
        self.good_samples_b_trans = np.empty((sample_count, self.K))

    def run(self, verbose: bool = False) -> None:
        for it in range(self.n_iter):
            omega = random_polyagamma(1, self.state.linear_predictor())
            _sample_weights(self.state, self.Data, self.sigma_w, omega=omega)
            _sample_biases(self.state, self.Data, self.sigma_b, self.mu_b, omega=omega)
            _sample_masks(self.state, self.Data, self.rho, self.sigma_w, omega=omega)
            _sample_latent_sequences(self.state, self.Data)
            _sample_transition_parameters(
                self.state,
                alpha=self.alpha,
                gamma=self.gamma,
                delta=self.delta,
            )

            self.samples_W[it] = self.state.W
            self.samples_A[it] = self.state.A
            self.samples_b[it] = self.state.b
            self.samples_Z[it] = self.state.Z
            self.samples_a[it] = self.state.a
            self.samples_b_trans[it] = self.state.b_trans

            if verbose and (it + 1) % 25 == 0:
                occupied = self.state.occupied_feature_count()
                print(f"Iteration {it + 1}: occupied={occupied}")

    def get_posterior_samples(self) -> None:
        valid_idx = np.arange(self.burn, self.n_iter)
        if self.n_subsample is None or self.n_subsample >= len(valid_idx):
            chosen = valid_idx
        else:
            chosen = np.random.choice(valid_idx, size=self.n_subsample, replace=False)

        self.good_samples_W = self.samples_W[chosen]
        self.good_samples_A = self.samples_A[chosen]
        self.good_samples_b = self.samples_b[chosen]
        self.good_samples_Z = self.samples_Z[chosen]
        self.good_samples_a = self.samples_a[chosen]
        self.good_samples_b_trans = self.samples_b_trans[chosen]

    def posterior_states(self) -> list[DynamicState]:
        return [
            DynamicState(
                W=self.good_samples_W[i].copy(),
                A=self.good_samples_A[i].copy(),
                b=self.good_samples_b[i].copy(),
                Z=self.good_samples_Z[i].copy(),
                a=self.good_samples_a[i].copy(),
                b_trans=self.good_samples_b_trans[i].copy(),
            )
            for i in range(len(self.good_samples_W))
        ]

    def posterior_predictive_gibbs(
        self,
        cond_obs: np.ndarray,
        n_z_samples: int = 256,
        seed: int = 0,
    ) -> float:
        states = self.posterior_states()
        if not states:
            raise ValueError("call get_posterior_samples before posterior_predictive_gibbs")

        rng = np.random.default_rng(seed)
        values = [
            monte_carlo_conditional_probability(
                state=state,
                cond_obs=np.asarray(cond_obs, dtype=float),
                n_samples=n_z_samples,
                rng=np.random.default_rng(int(rng.integers(1_000_000_000))),
            )
            for state in states
        ]
        return float(np.mean(values))


def snapshot_particles(particles: list[DynamicState], weights: np.ndarray) -> dict[str, object]:
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


def _latent_candidate_probabilities(state: DynamicState) -> np.ndarray:
    prev_state = state.Z[-1]
    return np.where(prev_state == 1, state.b_trans, state.a)


def append_trial_via_importance(
    particle: DynamicState,
    observation: np.ndarray,
    alpha: float,
    gamma: float,
    delta: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> float:
    probs = _latent_candidate_probabilities(particle)
    z_candidates = rng.binomial(1, _clip_probability(probs), size=(n_candidates, particle.K)).astype(np.int8)
    eta_candidates = z_candidates @ particle.effective_weights() + particle.b
    log_likelihood = _bernoulli_log_prob_matrix(observation, eta_candidates)
    log_predictive = float(logsumexp(log_likelihood) - np.log(n_candidates))

    posterior_weights = np.exp(log_likelihood - logsumexp(log_likelihood))
    chosen = int(rng.choice(n_candidates, p=posterior_weights))
    particle.Z = np.vstack([particle.Z, z_candidates[chosen]])

    _sample_transition_parameters(
        particle,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
    )
    return log_predictive


def monte_carlo_conditional_probability(
    state: DynamicState,
    cond_obs: np.ndarray,
    n_samples: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    rng = np.random.default_rng() if rng is None else rng
    cond_obs = np.asarray(cond_obs, dtype=float)
    if cond_obs.shape != (state.S - 1,):
        raise ValueError(f"cond_obs must have shape ({state.S - 1},), got {cond_obs.shape}")

    probs = _latent_candidate_probabilities(state)
    z_candidates = rng.binomial(1, _clip_probability(probs), size=(n_samples, state.K)).astype(np.int8)
    eta_candidates = z_candidates @ state.effective_weights() + state.b
    log_obs = _bernoulli_log_prob_matrix(cond_obs, eta_candidates[:, : state.S - 1])
    normalized = np.exp(log_obs - logsumexp(log_obs))
    us_prob = expit(eta_candidates[:, -1])
    return float(normalized @ us_prob)


def marginal_us_probability_from_particle(
    state: DynamicState,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    probs = _latent_candidate_probabilities(state)
    z_candidates = rng.binomial(1, _clip_probability(probs), size=(n_samples, state.K)).astype(np.int8)
    eta_candidates = z_candidates @ state.effective_weights() + state.b
    us_prob = expit(eta_candidates[:, -1])
    return float(np.mean(us_prob))


def _query_key(conds: list[int] | None) -> str:
    return "__marginal_us__" if conds is None else str(conds)


def query_probabilities_from_particle(
    state: DynamicState,
    conds_list: list[list[int] | None],
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    probs = _latent_candidate_probabilities(state)
    z_candidates = rng.binomial(1, _clip_probability(probs), size=(n_samples, state.K)).astype(np.int8)
    eta_candidates = z_candidates @ state.effective_weights() + state.b
    us_prob = expit(eta_candidates[:, -1])

    results: dict[str, float] = {}
    for conds in conds_list:
        key = _query_key(conds)
        if conds is None:
            results[key] = float(np.mean(us_prob))
            continue
        cond_obs = np.asarray(conds, dtype=float)
        if cond_obs.shape != (state.S - 1,):
            raise ValueError(f"cond_obs must have shape ({state.S - 1},), got {cond_obs.shape}")
        log_obs = _bernoulli_log_prob_matrix(cond_obs, eta_candidates[:, : state.S - 1])
        normalized = np.exp(log_obs - logsumexp(log_obs))
        results[key] = float(normalized @ us_prob)
    return results


def weighted_query_probabilities(
    particles: list[DynamicState],
    weights: np.ndarray,
    conds_list: list[list[int] | None],
    query_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    summaries = {_query_key(conds): 0.0 for conds in conds_list}
    for particle, particle_weight in zip(particles, weights):
        particle_summary = query_probabilities_from_particle(
            state=particle,
            conds_list=conds_list,
            n_samples=query_samples,
            rng=np.random.default_rng(int(rng.integers(1_000_000_000))),
        )
        for key, value in particle_summary.items():
            summaries[key] += float(particle_weight) * value
    return summaries


def rejuvenation_sweep(
    particle: DynamicState,
    data: np.ndarray,
    rho: float,
    alpha: float,
    gamma: float,
    delta: float,
    sigma_w: float,
    sigma_b: float,
    mu_b: np.ndarray,
) -> None:
    if data.shape[0] != particle.T:
        raise ValueError("particle latent rows and data rows must match during rejuvenation")

    omega = random_polyagamma(1, particle.linear_predictor())
    _sample_weights(particle, data, sigma_w, omega=omega)
    _sample_biases(particle, data, sigma_b, mu_b, omega=omega)
    _sample_masks(particle, data, rho, sigma_w, omega=omega)
    _sample_latent_sequences(particle, data)
    _sample_transition_parameters(
        particle,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
    )


def initialize_particles(
    base_data: np.ndarray,
    n_particles: int,
    *,
    K: int,
    alpha: float,
    rho: float,
    sigma_w: float,
    sigma_b: float,
    mu_b: list[float] | tuple[float, ...] | np.ndarray | None,
    n_iter: int,
    burn: int,
    gamma: float,
    delta: float,
    seed: int,
) -> tuple[list[DynamicState], DynamicIFHMMSampler]:
    np.random.seed(seed)
    sampler = DynamicIFHMMSampler(
        Data=base_data,
        K=K,
        rho=rho,
        alpha=alpha,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        mu_b=mu_b,
        n_iter=n_iter,
        burn=burn,
        n_subsample=n_particles,
        gamma=gamma,
        delta=delta,
    )
    sampler.run(verbose=False)
    sampler.get_posterior_samples()
    return sampler.posterior_states(), sampler


def scenario_phase_slices(
    latent_inhibition: bool,
    overshadowing: bool,
    final_phase_trials: int,
    baseline_start: int = 0,
    latent_inhibition_start: int = 50,
    final_phase_start: int | None = None,
) -> dict[str, slice]:
    phase_slices: dict[str, slice] = {}
    phase_slices["baseline"] = slice(baseline_start, latent_inhibition_start)
    if final_phase_start is None:
        final_phase_start = latent_inhibition_start + (50 if latent_inhibition else 0)
    if latent_inhibition:
        phase_slices["latent_inhibition"] = slice(latent_inhibition_start, final_phase_start)
    final_phase_name = "overshadowing" if overshadowing else "reinforced_a"
    phase_slices[final_phase_name] = slice(final_phase_start, final_phase_start + final_phase_trials)
    return phase_slices


def summarize_phase_dynamics(
    particles: list[DynamicState],
    weights: np.ndarray,
    phase_slices: dict[str, slice],
    top_features: int = 4,
    occupancy_threshold: int = 1,
) -> dict[str, object]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    phase_names = list(phase_slices.keys())
    n_particles = len(particles)

    slot_presence = np.zeros(top_features, dtype=float)
    slot_overall_occupancy = np.zeros(top_features, dtype=float)
    slot_a = np.zeros(top_features, dtype=float)
    slot_b_trans = np.zeros(top_features, dtype=float)
    slot_effective_weight = np.zeros((top_features, particles[0].S), dtype=float)
    phase_activation = {phase: np.zeros(top_features, dtype=float) for phase in phase_names}

    for idx in range(n_particles):
        particle = particles[idx]
        particle_weight = weights[idx]
        occupancies = particle.Z.sum(axis=0)
        active_idx = np.flatnonzero(occupancies >= occupancy_threshold)
        if active_idx.size == 0:
            continue

        ordered = active_idx[np.argsort(occupancies[active_idx])[::-1]][:top_features]
        slot_presence[: ordered.size] += particle_weight
        slot_overall_occupancy[: ordered.size] += particle_weight * (occupancies[ordered] / particle.T)
        slot_a[: ordered.size] += particle_weight * particle.a[ordered]
        slot_b_trans[: ordered.size] += particle_weight * particle.b_trans[ordered]
        slot_effective_weight[: ordered.size] += particle_weight * particle.effective_weights()[ordered]

        for phase_name, phase_slice in phase_slices.items():
            phase_Z = particle.Z[phase_slice]
            phase_len = max(phase_Z.shape[0], 1)
            phase_activation[phase_name][: ordered.size] += particle_weight * (phase_Z[:, ordered].sum(axis=0) / phase_len)

    return {
        "slot_presence_prob": slot_presence,
        "slot_overall_occupancy_mean": slot_overall_occupancy,
        "slot_a_mean": slot_a,
        "slot_b_mean": slot_b_trans,
        "slot_effective_weight_mean": slot_effective_weight,
        "phase_activation_mean": phase_activation,
        "phase_names": phase_names,
    }


def summarize_mode_aligned_features(
    particles: list[DynamicState],
    weights: np.ndarray,
    active_count_threshold: int = 5,
) -> dict[str, object]:
    if len(particles) == 0:
        raise ValueError("particles must be non-empty")

    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.full(len(weights), 1.0 / len(weights), dtype=float)
    else:
        weights = weights / weights.sum()

    active_counts = np.array(
        [np.sum(particle.Z.sum(axis=0) > active_count_threshold) for particle in particles],
        dtype=int,
    )
    eligible = np.flatnonzero(active_counts > 0)
    if eligible.size == 0:
        raise ValueError("No particles contain active features above the threshold.")

    active_count_values = active_counts[eligible]
    unique_counts, sample_counts = np.unique(active_count_values, return_counts=True)
    target_count = int(unique_counts[np.argmax(sample_counts)])
    candidates = eligible[active_counts[eligible] == target_count]
    reference_idx = int(candidates[np.argmax(weights[candidates])])
    sample_count = int(candidates.size)

    reference_particle = particles[reference_idx]
    ref_occupancy = reference_particle.Z.sum(axis=0)
    ref_active_idx = np.flatnonzero(ref_occupancy > active_count_threshold)
    ref_vectors = reference_particle.effective_weights()[ref_active_idx]

    selected_weights = weights[candidates]
    if selected_weights.sum() <= 0:
        selected_weights = np.full(candidates.size, 1.0 / candidates.size, dtype=float)
    else:
        selected_weights = selected_weights / selected_weights.sum()

    T = reference_particle.T
    slot_vector_sum = np.zeros_like(ref_vectors, dtype=float)
    slot_occupancy_sum = np.zeros(ref_active_idx.size, dtype=float)
    slot_activation_sum = np.zeros((T, ref_active_idx.size), dtype=float)
    slot_switch_sum = np.zeros((T, ref_active_idx.size), dtype=float)
    bias_sum = np.zeros(reference_particle.S, dtype=float)

    for particle_idx, particle_weight in zip(candidates, selected_weights):
        particle = particles[particle_idx]
        if particle_weight <= 0:
            continue

        occupancy = particle.Z.sum(axis=0)
        active_idx = np.flatnonzero(occupancy > active_count_threshold)
        if active_idx.size != target_count:
            continue

        vectors = particle.effective_weights()[active_idx]
        cost = np.linalg.norm(ref_vectors[:, None, :] - vectors[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        bias_sum += particle_weight * particle.b

        for row, col in zip(row_ind, col_ind):
            aligned_feature_idx = active_idx[col]
            aligned_z = particle.Z[:, aligned_feature_idx].astype(float)
            switch_indicator = np.zeros(T, dtype=float)
            switch_indicator[0] = aligned_z[0]
            if T > 1:
                switch_indicator[1:] = np.abs(np.diff(aligned_z))

            slot_vector_sum[row] += particle_weight * vectors[col]
            slot_occupancy_sum[row] += particle_weight * occupancy[aligned_feature_idx]
            slot_activation_sum[:, row] += particle_weight * aligned_z
            slot_switch_sum[:, row] += particle_weight * switch_indicator

    order = np.argsort(-slot_occupancy_sum)
    return {
        "target_count": target_count,
        "sample_count": sample_count,
        "reference_idx": reference_idx,
        "avg_vectors": slot_vector_sum[order],
        "avg_occupancy": slot_occupancy_sum[order],
        "avg_bias": bias_sum.tolist(),
        "avg_activation_by_time": slot_activation_sum[:, order],
        "avg_switch_prob_by_time": slot_switch_sum[:, order],
        "avg_switch_count_by_time": slot_switch_sum[:, order].sum(axis=1),
        "candidate_indices": candidates.tolist(),
        "active_counts": active_counts.tolist(),
    }


def summarize_phase_switches(
    aligned_summary: dict[str, object],
    phase_slices: dict[str, slice],
    switch_prob_threshold: float = 0.25,
    activation_delta_threshold: float = 0.1,
) -> dict[str, object]:
    switch_count = np.asarray(aligned_summary["avg_switch_count_by_time"], dtype=float)
    switch_prob = np.asarray(aligned_summary["avg_switch_prob_by_time"], dtype=float)
    activation = np.asarray(aligned_summary["avg_activation_by_time"], dtype=float)
    T = switch_count.size

    phase_items = sorted(phase_slices.items(), key=lambda item: item[1].start)
    phase_indicator = np.zeros(T, dtype=float)
    transition_reports = []
    phase_mean_switch_count: dict[str, float] = {}
    phase_switch_correlation: dict[str, float] = {}

    for phase_name, phase_slice in phase_items:
        indicator = np.zeros(T, dtype=float)
        indicator[phase_slice] = 1.0
        phase_mean_switch_count[phase_name] = float(np.mean(switch_count[phase_slice]))
        if np.std(indicator) > 0 and np.std(switch_count) > 0:
            phase_switch_correlation[phase_name] = float(np.corrcoef(switch_count, indicator)[0, 1])
        else:
            phase_switch_correlation[phase_name] = np.nan

    for idx in range(1, len(phase_items)):
        from_phase, from_slice = phase_items[idx - 1]
        to_phase, to_slice = phase_items[idx]
        transition_t = int(to_slice.start)
        if transition_t <= 0 or transition_t >= T:
            continue

        phase_indicator[transition_t] = 1.0
        switch_probs_t = switch_prob[transition_t]
        activation_delta = activation[transition_t] - activation[transition_t - 1]
        feature_rows = []
        interesting = np.flatnonzero(
            (switch_probs_t >= switch_prob_threshold)
            | (np.abs(activation_delta) >= activation_delta_threshold)
        )

        for feature_idx in interesting:
            delta_value = float(activation_delta[feature_idx])
            if delta_value > activation_delta_threshold:
                direction = "on"
            elif delta_value < -activation_delta_threshold:
                direction = "off"
            else:
                direction = "mixed"
            feature_rows.append({
                "slot": int(feature_idx + 1),
                "switch_prob": float(switch_probs_t[feature_idx]),
                "activation_before": float(activation[transition_t - 1, feature_idx]),
                "activation_after": float(activation[transition_t, feature_idx]),
                "activation_delta": delta_value,
                "direction": direction,
            })

        transition_reports.append({
            "from_phase": from_phase,
            "to_phase": to_phase,
            "time_index": transition_t,
            "mean_switch_count": float(switch_count[transition_t]),
            "features": feature_rows,
        })

    if np.std(phase_indicator) > 0 and np.std(switch_count) > 0:
        boundary_correlation = float(np.corrcoef(switch_count, phase_indicator)[0, 1])
    else:
        boundary_correlation = np.nan

    return {
        "switch_count_by_time": switch_count,
        "phase_mean_switch_count": phase_mean_switch_count,
        "phase_switch_correlation": phase_switch_correlation,
        "boundary_indicator": phase_indicator,
        "boundary_correlation": boundary_correlation,
        "transition_reports": transition_reports,
    }


def run_dynamic_li_ov_particle_filter(
    final_phase_trials: int = 50,
    *,
    acquisition_trials: int | None = None,
    latent_inhibition: bool = True,
    overshadowing: bool = True,
    seed: int = 45,
    n_particles: int = 64,
    K: int = 16,
    alpha: float = 1.0,
    gamma: float = 6.0,
    delta: float = 1.0,
    rho: float = 0.25,
    sigma_w: float = 2.0,
    sigma_b: float = 1.0,
    mu_b: list[float] | tuple[float, ...] | np.ndarray | None = None,
    init_n_iter: int = 2000,
    init_burn: int = 1000,
    latent_update_candidates: int = 256,
    query_samples: int = 256,
    ess_threshold_fraction: float = 0.5,
    rejuvenation_sweeps_per_resample: int = 1,
    conds_list: list[list[int] | None] | None = None,
    snapshot_trials: list[int] | tuple[int, ...] | None = None,
    verbose: bool = True,
    progress_every: int = 5,
    run_label: str | None = None,
    init_verbose: bool = False,
) -> dict[str, object]:
    if acquisition_trials is not None:
        final_phase_trials = acquisition_trials

    rng = np.random.default_rng(seed)
    label = run_label or f"seed={seed}"
    full_data = _safe_generate_li_ov_data(
        trials=final_phase_trials,
        latent_inhibition=latent_inhibition,
        overshadowing=overshadowing,
    )
    base_data = full_data[:-final_phase_trials]
    final_phase_data = full_data[-final_phase_trials:]
    final_phase_name = "overshadowing" if overshadowing else "reinforced_a"
    if verbose:
        print(
            f"[{label}] Initial posterior fit on {base_data.shape[0]} prefix trials "
            f"(LI={latent_inhibition}, OV={overshadowing})",
            flush=True,
        )
    particles, base_sampler = initialize_particles(
        base_data=base_data,
        n_particles=n_particles,
        K=K,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        rho=rho,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        mu_b=mu_b,
        n_iter=init_n_iter,
        burn=init_burn,
        seed=seed,
    )
    if verbose:
        print(f"[{label}] Initial fit complete. Starting sequential filter.", flush=True)

    if conds_list is None:
        conds_list = [list(conds) for conds in DEFAULT_LI_OV_CONDITIONS]

    n_particles = len(particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    posterior_probs = {_query_key(conds): [] for conds in conds_list}
    snapshot_trials = set(snapshot_trials or [])
    snapshots: dict[int, dict[str, object]] = {}

    initial_query = weighted_query_probabilities(
        particles=particles,
        weights=weights,
        conds_list=conds_list,
        query_samples=query_samples,
        rng=rng,
    )
    for conds in conds_list:
        key = _query_key(conds)
        posterior_probs[key].append(initial_query[key])
    if 0 in snapshot_trials:
        snapshots[0] = snapshot_particles(particles, weights)

    ess_history = [float(n_particles)]
    ess_before_resample_history = [float(n_particles)]
    occupied_history = [float(weights @ np.array([p.occupied_feature_count() for p in particles], dtype=float))]
    resampled_at: list[int] = []
    data_so_far = base_data.copy()

    for trial_count, observation in enumerate(final_phase_data, start=1):
        log_weights = np.log(weights + 1e-300)
        for idx, particle in enumerate(particles):
            log_predictive = append_trial_via_importance(
                particle=particle,
                observation=observation,
                alpha=alpha,
                gamma=gamma,
                delta=delta,
                n_candidates=latent_update_candidates,
                rng=rng,
            )
            log_weights[idx] += log_predictive

        data_so_far = np.vstack([data_so_far, observation])
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        ess = float(1.0 / np.sum(weights**2))
        ess_before_resample = ess

        if ess < ess_threshold_fraction * n_particles:
            ancestor_idx = systematic_resample(weights, rng)
            particles = [particles[i].copy() for i in ancestor_idx]
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
            mu_b_arr = np.full(data_so_far.shape[1], -3.0, dtype=float) if mu_b is None else np.asarray(mu_b, dtype=float)
            for particle in particles:
                for _ in range(rejuvenation_sweeps_per_resample):
                    rejuvenation_sweep(
                        particle=particle,
                        data=data_so_far,
                        rho=rho,
                        alpha=alpha,
                        gamma=gamma,
                        delta=delta,
                        sigma_w=sigma_w,
                        sigma_b=sigma_b,
                        mu_b=mu_b_arr,
                    )
            resampled_at.append(trial_count)
            ess = float(n_particles)
            if verbose:
                print(f"[{label}] Resampled/rejuvenated at final-phase trial {trial_count}.", flush=True)

        query_summary = weighted_query_probabilities(
            particles=particles,
            weights=weights,
            conds_list=conds_list,
            query_samples=query_samples,
            rng=rng,
        )
        for conds in conds_list:
            key = _query_key(conds)
            posterior_probs[key].append(query_summary[key])

        ess_history.append(ess)
        ess_before_resample_history.append(ess_before_resample)
        occupied_history.append(float(weights @ np.array([p.occupied_feature_count() for p in particles], dtype=float)))

        if trial_count in snapshot_trials:
            snapshots[trial_count] = snapshot_particles(particles, weights)

        if verbose and (
            trial_count == 1
            or trial_count == final_phase_trials
            or (progress_every > 0 and trial_count % progress_every == 0)
        ):
            print(
                f"[{label}] Filter progress {trial_count}/{final_phase_trials} "
                f"(ESS={ess:.1f}, mean occupied={occupied_history[-1]:.2f})",
                flush=True,
            )

    return {
        "trial_counts": list(range(final_phase_trials + 1)),
        "acquisition_values": list(range(final_phase_trials + 1)),
        "posterior_probs": posterior_probs,
        "ess_history": ess_history,
        "ess_before_resample_history": ess_before_resample_history,
        "occupied_history": occupied_history,
        "resampled_at": resampled_at,
        "snapshots": snapshots,
        "conds_list": conds_list,
        "cond_labels": DEFAULT_LI_OV_CONDITION_LABELS.copy(),
        "base_sampler": base_sampler,
        "settings": {
            "seed": seed,
            "latent_inhibition": latent_inhibition,
            "overshadowing": overshadowing,
            "final_phase_name": final_phase_name,
            "final_phase_trials": final_phase_trials,
            "n_particles": n_particles,
            "K": K,
            "alpha": alpha,
            "gamma": gamma,
            "delta": delta,
            "rho": rho,
            "sigma_w": sigma_w,
            "sigma_b": sigma_b,
            "mu_b": None if mu_b is None else list(np.asarray(mu_b, dtype=float)),
            "init_n_iter": init_n_iter,
            "init_burn": init_burn,
            "latent_update_candidates": latent_update_candidates,
            "query_samples": query_samples,
            "ess_threshold_fraction": ess_threshold_fraction,
            "rejuvenation_sweeps_per_resample": rejuvenation_sweeps_per_resample,
        },
    }


def compare_li_ov_predictions(
    final_phase_trials: int = 50,
    *,
    n_runs: int = 4,
    seeds: list[int] | None = None,
    query_condition: list[int] | tuple[int, int] | None = (1, 0),
    scenario_order: tuple[dict[str, object], ...] = LI_OV_SCENARIOS,
    verbose: bool = True,
    **kwargs,
) -> dict[str, object]:
    if seeds is None:
        seeds = [45 + idx for idx in range(n_runs)]
    if len(seeds) != n_runs:
        raise ValueError("len(seeds) must match n_runs")

    scenario_results: dict[str, list[dict[str, object]]] = {}
    conds = None if query_condition is None else list(query_condition)
    key = _query_key(conds)
    run_kwargs = dict(kwargs)
    conds_list = run_kwargs.pop("conds_list", None)
    if conds_list is None:
        conds_list = [conds]
    else:
        conds_list = list(conds_list)
        if conds is None:
            if not any(item is None for item in conds_list):
                conds_list.append(None)
        else:
            normalized = [None if item is None else list(item) for item in conds_list]
            if conds not in [item for item in normalized if item is not None]:
                conds_list.append(conds)

    for scenario_idx, scenario in enumerate(scenario_order, start=1):
        runs = []
        for run_idx, seed in enumerate(seeds, start=1):
            if verbose:
                print(
                    f"[scenario {scenario_idx}/{len(scenario_order)} | run {run_idx}/{n_runs}] "
                    f"{scenario['label']} seed={seed}",
                    flush=True,
                )
            run = run_dynamic_li_ov_particle_filter(
                final_phase_trials=final_phase_trials,
                latent_inhibition=bool(scenario["latent_inhibition"]),
                overshadowing=bool(scenario["overshadowing"]),
                seed=seed,
                verbose=verbose,
                run_label=f"{scenario['name']} run {run_idx}/{n_runs} seed={seed}",
                conds_list=conds_list,
                **run_kwargs,
            )
            runs.append(run)
        scenario_results[scenario["name"]] = runs

    trial_counts = np.asarray(next(iter(scenario_results.values()))[0]["trial_counts"], dtype=int)
    summary = {}
    for scenario in scenario_order:
        name = scenario["name"]
        curves = np.asarray([run["posterior_probs"][key] for run in scenario_results[name]], dtype=float)
        summary[name] = {
            "label": scenario["label"],
            "mean": curves.mean(axis=0),
            "std": curves.std(axis=0),
            "runs": curves,
        }

    return {
        "trial_counts": trial_counts,
        "acquisition_values": trial_counts,
        "query_condition": conds,
        "scenario_results": scenario_results,
        "summary": summary,
        "scenario_order": list(scenario_order),
    }


def compare_li_ov_acquisition(
    acquisition_trials: int = 50,
    *,
    n_runs: int = 4,
    seeds: list[int] | None = None,
    query_condition: list[int] | tuple[int, int] | None = (1, 0),
    scenario_order: tuple[dict[str, object], ...] = LI_OV_SCENARIOS,
    verbose: bool = True,
    **kwargs,
) -> dict[str, object]:
    return compare_li_ov_predictions(
        final_phase_trials=acquisition_trials,
        n_runs=n_runs,
        seeds=seeds,
        query_condition=query_condition,
        scenario_order=scenario_order,
        verbose=verbose,
        **kwargs,
    )
