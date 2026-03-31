import math

import numpy as np
from scipy.special import expit
from polyagamma import random_polyagamma


def _bernoulli_log_prob_from_eta(y, eta):
    return np.sum(y * eta - np.logaddexp(0.0, eta))


def _clip_probability(p, eps=1e-12):
    return float(np.clip(p, eps, 1.0 - eps))


def _geometric_mask_count_probs(S, rho):
    """
    Truncated geometric prior over mask cardinality m in {0, ..., S}.

    The parameter rho is the geometric ratio so the unnormalized law is

        P(M = m) propto rho ** m

    with subsets of a given size treated uniformly. This keeps rho in (0, 1),
    with smaller values preferring sparser masks and larger values allowing
    denser masks.
    """

    rho = _clip_probability(rho)
    support = np.arange(S + 1, dtype=float)
    weights = np.power(rho, support)
    return weights / weights.sum()


def _sample_geometric_mask_row(S, rho):
    probs = _geometric_mask_count_probs(S, rho)
    count = int(np.random.choice(np.arange(S + 1), p=probs))
    row = np.zeros(S, dtype=np.int8)
    if count > 0:
        active_idx = np.random.choice(S, size=count, replace=False)
        row[active_idx] = 1
    return row


def _geometric_mask_log_prob(row, rho):
    row = np.asarray(row, dtype=np.int8)
    S = row.size
    probs = _geometric_mask_count_probs(S, rho)
    count = int(row.sum())
    return np.log(probs[count]) - math.log(math.comb(S, count))

# ============================================================
# Forward Model
# ============================================================

class ParametricLLFM:
    """
    Finite-K logistic latent feature model with
    Beta-Bernoulli prior (IBP finite approximation):

        π_k ~ Beta(alpha/K, 1)
        z_{t,k} ~ Bernoulli(π_k)

        y_{t,d} ~ Bernoulli(sigmoid(Z_t W_d + b_d))

    Z : (T × K)
    W : (K × S)
    b : (S,)
    π : (K,)
    """

    def __init__(self, T, S, K, rho, alpha=1, mu_b=[-1.0]*4,
                 sigma_w=3, sigma_b=1.0):
        """
        Parameters
        ----------
        fixed_bias : None or array of length S. If not None, biases are fixed to this value.
        """
        self.T = T
        self.S = S
        self.K = K
        self.rho = rho
        self.alpha = alpha
        self.mu_b = np.array(mu_b)
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        # Beta-Bernoulli prior for latent features
        self.pi = np.random.beta(self.alpha / K, 1.0, size=K)
        self.Z = np.random.binomial(1, self.pi, size=(T, K))
        self.W = np.random.normal(0, self.sigma_w, size=(K, S))
        self.A = np.random.binomial(1, rho, size=(K, S))
        self.b = np.random.normal(self.mu_b, self.sigma_b, size=S)

    # -----------------------------------------------------

    def linear_predictor(self):
        return self.Z @ (self.A * self.W) + self.b  # (T × S)

    def sigmoid(self):
        return expit(self.linear_predictor())

    def sample_observations(self):
        P = self.sigmoid()
        return np.random.binomial(1, P)  # (T × S)


class ParametricLLFMGeometricMask(ParametricLLFM):
    """
    Finite-K logistic latent feature model with the same latent-feature prior as
    ParametricLLFM, but where each feature mask row is sampled from a truncated
    geometric distribution over the number of active dimensions rather than
    entrywise Bernoulli draws.
    """

    def __init__(self, T, S, K, rho, alpha=1, mu_b=[-1.0] * 4,
                 sigma_w=3, sigma_b=1.0):
        super().__init__(
            T=T,
            S=S,
            K=K,
            rho=rho,
            alpha=alpha,
            mu_b=mu_b,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
        )
        self.A = np.zeros((K, S), dtype=np.int8)
        self.W = np.zeros((K, S))
        for k in range(K):
            self.A[k] = _sample_geometric_mask_row(S, rho)
            active_idx = np.flatnonzero(self.A[k])
            if active_idx.size > 0:
                self.W[k, active_idx] = np.random.normal(0, self.sigma_w, size=active_idx.size)


# ============================================================
# Gibbs Sampler with Polya-Gamma augmentation
# ============================================================

class GibbsSamplerLLFM:
    """
    Collapsed Gibbs sampler for finite-K logistic latent feature model
    with Polya–Gamma augmentation and Beta-Bernoulli prior
    (π integrated out).
    """

    def __init__(self, Data, K=15, rho=0.1, alpha=0.5, sigma_w=3.0, sigma_b=1.0, mu_b=[-1.0]*4,
                 n_iter=1000, burn=200, n_subsample=None):
        self.Data = Data
        self.T, self.S = Data.shape
        self.K = K
        self.alpha = alpha
        self.rho = rho
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.mu_b = np.array(mu_b)
        # Base model
        self.model = self._make_model()

        # Gibbs sampling parameters
        self.n_iter = n_iter
        self.burn = burn
        self.n_subsample = n_subsample

        # Storage
        self.samples_W = np.empty((self.n_iter, self.K, self.S))
        self.samples_b = np.empty((self.n_iter, self.S))
        self.samples_Z = np.empty((self.n_iter, self.T, self.K), dtype=np.int8)
        self.samples_A = np.empty((self.n_iter, self.K, self.S), dtype=np.int8)
        sample_count = 0 if self.n_subsample is None else self.n_subsample
        self.good_samples_W = np.empty((sample_count, self.K, self.S))
        self.good_samples_b = np.empty((sample_count, self.S))
        self.good_samples_Z = np.empty((sample_count, self.T, self.K), dtype=np.int8)
        self.good_samples_A = np.empty((sample_count, self.K, self.S), dtype=np.int8)

    def _make_model(self):
        return ParametricLLFM(
            T=self.T,
            S=self.S,
            K=self.K,
            rho=self.rho,
            alpha=self.alpha,
            mu_b=self.mu_b,
            sigma_w=self.sigma_w,
            sigma_b=self.sigma_b,
        )

    def _row_mask_log_prob(self, row):
        row = np.asarray(row, dtype=np.int8)
        rho = _clip_probability(self.rho)
        return (
            row.sum() * np.log(rho)
            + (row.size - row.sum()) * np.log(1.0 - rho)
        )

    # --------------------------------------------------------
    # Main Gibbs loop
    # --------------------------------------------------------

    def run(self, verbose=False):
        for it in range(self.n_iter):

            # 1) Sample Polya-Gamma variables
            eta = self.model.linear_predictor()
            omega = random_polyagamma(1, eta)

            # 2) Sample W
            for s in range(self.S):
                Z = self.model.Z
                a = self.model.A[:, s]
                active_idx = np.flatnonzero(a)
                if active_idx.size == 0:
                    self.model.W[:, s] = 0.0
                    continue

                omega_s = omega[:, s]
                Z_active = Z[:, active_idx]
                kappa = self.Data[:, s] - 0.5
                ZOmega = omega_s[:, None] * Z_active
                V_inv = Z_active.T @ ZOmega + np.eye(active_idx.size) / self.sigma_w**2
                b_s = self.model.b[s]
                rhs = Z_active.T @ (kappa - omega_s * b_s)
                L = np.linalg.cholesky(V_inv)
                mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
                noise = np.linalg.solve(L.T, np.random.randn(active_idx.size))
                w_new = mu + noise
                self.model.W[:, s] = 0.0
                self.model.W[active_idx, s] = w_new

            # 3) Sample b
            for s in range(self.S):
                omega_s = omega[:, s]
                kappa = self.Data[:, s] - 0.5
                linear_part = self.model.Z @ (self.model.A[:, s] * self.model.W[:, s])
                V_b = 1.0 / (omega_s.sum() + 1.0 / self.sigma_b**2)
                m_bs = V_b * (np.sum(kappa - omega_s * linear_part) + self.mu_b[s] / self.sigma_b**2)
                self.model.b[s] = m_bs + np.sqrt(V_b) * np.random.randn()

            # 4) Sample A jointly with its slab weight under the PG-augmented model.
            # This allows inactive edges to turn on due to the likelihood.
            for s in range(self.S):
                omega_s = omega[:, s]
                kappa = self.Data[:, s] - 0.5
                eta_s = self.model.Z @ (self.model.A[:, s] * self.model.W[:, s]) + self.model.b[s]

                for k in range(self.K):
                    z = self.model.Z[:, k]
                    old_a = self.model.A[k, s]
                    old_w = self.model.W[k, s]
                    current_contrib = z * (old_a * old_w)
                    eta_minus = eta_s - current_contrib

                    x_omega_x = np.dot(omega_s, z)
                    precision = x_omega_x + 1.0 / self.sigma_w**2
                    rhs = np.dot(z, kappa - omega_s * eta_minus)

                    row_base = self.model.A[k].copy()
                    row_base[s] = 0
                    row_on = row_base.copy()
                    row_on[s] = 1
                    row_off = row_base

                    logp1 = (
                        self._row_mask_log_prob(row_on)
                        - 0.5 * np.log(precision * self.sigma_w**2)
                        + 0.5 * (rhs**2) / precision
                    )
                    logp0 = self._row_mask_log_prob(row_off)

                    prob = expit(logp1 - logp0)
                    a_new = np.random.binomial(1, prob)

                    if a_new == 1:
                        var_w = 1.0 / precision
                        mean_w = rhs * var_w
                        w_new = mean_w + np.sqrt(var_w) * np.random.randn()
                    else:
                        w_new = 0.0

                    self.model.A[k, s] = a_new
                    self.model.W[k, s] = w_new
                    eta_s = eta_minus + z * (a_new * w_new)

            # 5) Collapsed Z update
            W_eff = self.model.A * self.model.W
            for t in range(self.T):

                eta_t = self.model.Z[t] @ W_eff + self.model.b

                for k in range(self.K):

                    z_old = self.model.Z[t, k]

                    m_k = self.model.Z[:, k].sum() - z_old

                    log_prior1 = np.log(m_k + self.alpha / self.K + 1e-12)
                    log_prior0 = np.log(self.T - m_k + 1e-12)

                    contrib = W_eff[k]

                    eta_minus = eta_t - contrib if z_old else eta_t

                    # z=0
                    ll0 = _bernoulli_log_prob_from_eta(self.Data[t], eta_minus)

                    # z=1
                    eta_plus = eta_minus + contrib
                    ll1 = _bernoulli_log_prob_from_eta(self.Data[t], eta_plus)

                    logp0 = ll0 + log_prior0
                    logp1 = ll1 + log_prior1

                    prob = expit(logp1 - logp0)

                    z_new = np.random.binomial(1, prob)

                    self.model.Z[t, k] = z_new

                    if z_new != z_old:
                        eta_t = eta_minus + z_new * contrib


            # Store samples
            self.samples_W[it] = self.model.W
            self.samples_A[it] = self.model.A
            self.samples_b[it] = self.model.b
            self.samples_Z[it] = self.model.Z

            if verbose and (it+1) % 25 == 0:
                fro_W = np.linalg.norm(self.model.W, ord='fro')
                print(f"Iteration {it+1}: ||W||_F = {fro_W:.4f}")

    # --------------------------------------------------------
    # Subsample posterior
    # --------------------------------------------------------
    def get_posterior_samples(self):
        burn = self.burn
        n_subsample = self.n_subsample
        total = len(self.samples_W)
        valid_idx = np.arange(burn, total)
        if n_subsample is None or n_subsample >= len(valid_idx):
            chosen = valid_idx
        else:
            chosen = np.random.choice(valid_idx, size=n_subsample, replace=False)

        self.good_samples_W = self.samples_W[chosen]
        self.good_samples_b = self.samples_b[chosen]
        self.good_samples_Z = self.samples_Z[chosen]
        self.good_samples_A = self.samples_A[chosen]
        #return self.good_samples_W, self.good_samples_b, self.good_samples_Z, self.good_samples_A
    

    def posterior_predictive_gibbs(self, cond_obs,
                               n_z_samples=50,
                               burn_z=20,
                               gibbs_steps=1):

        cond_obs = np.asarray(cond_obs, dtype=float)

        N, K, S = self.good_samples_W.shape
        if cond_obs.shape != (S - 1,):
            raise ValueError(f"cond_obs must have shape ({S - 1},), got {cond_obs.shape}")

        gibbs_steps = max(1, int(gibbs_steps))
        probs = np.empty(N)

        for n in range(N):

            W = self.good_samples_W[n]
            A = self.good_samples_A[n]
            b = self.good_samples_b[n]
            Z_post = self.good_samples_Z[n]

            m = Z_post.sum(axis=0)
            active_idx = np.flatnonzero(m > 0)
            if active_idx.size == 0:
                probs[n] = expit(b[-1])
                continue

            W_eff = (A * W)[active_idx]
            m_active = m[active_idx]

            pi = (m_active + self.alpha / self.K) / (self.T + 1.0 + self.alpha / self.K)
            pi = np.clip(pi, 1e-12, 1.0 - 1e-12)

            # initialize once
            z = np.random.binomial(1, pi)
            eta = z @ W_eff + b

            total_steps = burn_z + n_z_samples
            us_prob_sum = 0.0
            kept_samples = 0

            for step in range(total_steps):

                for _ in range(gibbs_steps):
                    for k in range(active_idx.size):

                        z_old = z[k]
                        w = W_eff[k]

                        eta_minus = eta - z_old * w

                        eta_obs0 = eta_minus[:S - 1]
                        ll0 = _bernoulli_log_prob_from_eta(cond_obs, eta_obs0) + np.log(1.0 - pi[k])

                        eta_plus = eta_minus + w
                        eta_obs1 = eta_plus[:S - 1]
                        ll1 = _bernoulli_log_prob_from_eta(cond_obs, eta_obs1) + np.log(pi[k])

                        p = expit(ll1 - ll0)

                        z_new = np.random.binomial(1, p)
                        z[k] = z_new

                        eta = eta_minus + z_new * w

                # collect samples after burn-in
                if step >= burn_z:
                    us_prob_sum += expit(eta[-1])
                    kept_samples += 1

            probs[n] = us_prob_sum / kept_samples

        return probs.mean()


class GibbsSamplerLLFMGeometricMask(GibbsSamplerLLFM):
    """
    Same Gibbs sampler as GibbsSamplerLLFM, but with a geometric prior over the
    number of active weights in each feature mask row.
    """

    def _make_model(self):
        return ParametricLLFMGeometricMask(
            T=self.T,
            S=self.S,
            K=self.K,
            rho=self.rho,
            alpha=self.alpha,
            mu_b=self.mu_b,
            sigma_w=self.sigma_w,
            sigma_b=self.sigma_b,
        )

    def _row_mask_log_prob(self, row):
        return _geometric_mask_log_prob(row, self.rho)
