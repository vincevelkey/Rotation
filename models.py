import numpy as np
from scipy.special import expit, logsumexp  # sigmoid and stable logsumexp
from polyagamma import random_polyagamma

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

    def __init__(self, T, S, K, alpha=0.5, mu_b=-15.0,
                 sigma_w=3, sigma_b=1.0, fixed_bias=None):
        """
        Parameters
        ----------
        fixed_bias : None or array of length S. If not None, biases are fixed to this value.
        """
        self.T = T
        self.S = S
        self.K = K
        self.alpha = alpha
        self.mu_b = mu_b
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        # Beta-Bernoulli prior for latent features
        self.pi = np.random.beta(alpha / K, 1.0, size=K)
        self.Z = np.random.binomial(1, self.pi, size=(T, K))
        self.W = np.random.normal(0, sigma_w, size=(K, S))

        # Fixed bias option
        if fixed_bias is not None:
            assert len(fixed_bias) == S, "fixed_bias must have length S"
            self.b = np.array(fixed_bias)
            self.fixed_bias = True
        else:
            self.b = np.random.normal(0, sigma_b, size=(S,))
            self.fixed_bias = False

    # -----------------------------------------------------

    def linear_predictor(self):
        return self.Z @ self.W + self.b  # (T × S)

    def sigmoid(self):
        return expit(self.linear_predictor())

    def sample_observations(self):
        P = self.sigmoid()
        return np.random.binomial(1, P)  # (T × S)


# ============================================================
# Gibbs Sampler with Polya-Gamma augmentation
# ============================================================

class GibbsSamplerLLFM:
    """
    Collapsed Gibbs sampler for finite-K logistic latent feature model
    with Polya–Gamma augmentation and Beta-Bernoulli prior
    (π integrated out). Optionally allows fixed bias.
    """

    def __init__(self, Data, K=15, alpha=0.5, sigma_w=3.0, sigma_b=1.0, mu_b=-15.0,
                 n_iter=1000, burn=200, n_subsample=None, fixed_bias=None):
        self.Data = Data
        self.T, self.S = Data.shape
        self.K = K
        self.alpha = alpha
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.mu_b = mu_b
        self.fixed_bias = fixed_bias

        # Base model
        self.model = ParametricLLFM(self.T, self.S, self.K, self.alpha,
                                    self.mu_b, self.sigma_w, self.sigma_b,
                                    fixed_bias=self.fixed_bias)

        # Gibbs sampling parameters
        self.n_iter = n_iter
        self.burn = burn
        self.n_subsample = n_subsample

        # Storage
        self.samples_W = np.empty((self.n_iter, self.K, self.S))
        self.samples_b = np.empty((self.n_iter, self.S))
        self.samples_Z = np.empty((self.n_iter, self.T, self.K))
        self.good_samples_W = np.empty((self.n_subsample, self.K, self.S))
        self.good_samples_b = np.empty((self.n_subsample, self.S))
        self.good_samples_Z = np.empty((self.n_subsample, self.T, self.K))

    # --------------------------------------------------------
    # Main Gibbs loop
    # --------------------------------------------------------

    def run(self, verbose=False):
        for it in range(self.n_iter):
            if verbose:
                print(f"Iteration {it+1}/{self.n_iter}")

            # 1) Sample Polya-Gamma variables
            eta = self.model.linear_predictor()
            omega = random_polyagamma(1, eta)

            # 2) Sample W
            for s in range(self.S):
                Z = self.model.Z
                omega_d = omega[:, s]
                kappa = self.Data[:, s] - 0.5
                Omega = np.diag(omega_d)
                V_inv = Z.T @ Omega @ Z + np.eye(self.K) / self.sigma_w**2
                rhs = Z.T @ kappa
                L = np.linalg.cholesky(V_inv)
                mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
                noise = np.linalg.solve(L.T, np.random.randn(self.K))
                self.model.W[:, s] = mu + noise

            # 3) Sample or fix bias
            if not self.model.fixed_bias:
                for s in range(self.S):
                    omega_d = omega[:, s]
                    kappa = self.Data[:, s] - 0.5
                    linear_part = self.model.Z @ self.model.W[:, s]
                    V_b = 1.0 / (omega_d.sum() + 1.0 / self.sigma_b**2)
                    m_b = V_b * (np.sum(kappa - omega_d * linear_part) + self.mu_b / self.sigma_b**2)
                    self.model.b[s] = m_b + np.sqrt(V_b) * np.random.randn()
            else:
                # fixed bias, do nothing (already set in model)
                pass

            # 4) Collapsed Z update
            for t in range(self.T):
                eta_t = self.model.Z[t] @ self.model.W + self.model.b
                for k in range(self.K):
                    z_old = self.model.Z[t, k]
                    m_k = self.model.Z[:, k].sum() - z_old
                    log_prior_1 = np.log(m_k + self.alpha / self.K + 1e-12)
                    log_prior_0 = np.log(self.T - m_k + 1e-12)
                    eta_minus = eta_t - self.model.W[k] if z_old == 1 else eta_t

                    # z = 0
                    ll0 = np.sum(
                        self.Data[t] * np.log(expit(eta_minus)+1e-12) +
                        (1 - self.Data[t]) * np.log(1 - expit(eta_minus)+1e-12)
                    )
                    # z = 1
                    eta_plus = eta_minus + self.model.W[k]
                    ll1 = np.sum(
                        self.Data[t] * np.log(expit(eta_plus)+1e-12) +
                        (1 - self.Data[t]) * np.log(1 - expit(eta_plus)+1e-12)
                    )

                    logp0 = ll0 + log_prior_0
                    logp1 = ll1 + log_prior_1
                    p1 = 1.0 / (1.0 + np.exp(-(logp1 - logp0)))
                    z_new = np.random.binomial(1, p1)
                    self.model.Z[t, k] = z_new
                    if z_new != z_old:
                        eta_t = eta_minus + z_new * self.model.W[k]

            # Store samples
            self.samples_W[it] = self.model.W
            self.samples_b[it] = self.model.b
            self.samples_Z[it] = self.model.Z

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
        return self.good_samples_W, self.good_samples_b, self.good_samples_Z

    # --------------------------------------------------------
    # Posterior predictive (vectorized, last dimension)
    # --------------------------------------------------------
    def posterior_predictive_vectorised(self, cond_obs, n_z_samples=50):
        """
        Predict P(Y_last | Y_0:S-2 = cond_obs, data)
        cond_obs: array of shape (S-1,), binary assignment of observed features
        """
        samples_W = self.good_samples_W
        samples_b = self.good_samples_b
        samples_Z = self.good_samples_Z
        cond_obs = np.array(cond_obs)
        N, K, S = samples_W.shape
        assert len(cond_obs) == S-1, "cond_obs must have length S-1"

        # ----- Sample latent Z for Monte Carlo -----
        Z_train_counts = samples_Z.sum(axis=1)
        prior_prob = Z_train_counts / self.T
        Z_new = np.random.binomial(1, prior_prob[:, None, :], size=(N, n_z_samples, K))

        # ----- Expand W and b -----
        W_expand = samples_W[:, None, :, :]
        W_expand = np.repeat(W_expand, n_z_samples, axis=1)
        b_expand = samples_b[:, None, :]
        b_expand = np.repeat(b_expand, n_z_samples, axis=1)

        # ----- Full obs: conditioned + prediction -----
        obs = np.concatenate([cond_obs, [1]])  # last = predict 1
        indices = list(range(S))
        eta_all = np.einsum('nzk,nzks->nzs', Z_new, W_expand[:, :, :, indices]) + b_expand[:, :, indices]

        obs_expand = obs[None, None, :]
        logp_all = obs_expand * np.log(expit(eta_all)+1e-12) + (1-obs_expand) * np.log(1-expit(eta_all)+1e-12)
        logp_all = np.sum(logp_all, axis=2)

        eta_cond = eta_all[:, :, :S-1]
        obs_cond = cond_obs[None, None, :]
        logp_cond = obs_cond * np.log(expit(eta_cond)+1e-12) + (1-obs_cond) * np.log(1-expit(eta_cond)+1e-12)
        logp_cond = np.sum(logp_cond, axis=2)

        log_num = logsumexp(logp_all, axis=1) - np.log(n_z_samples)
        log_den = logsumexp(logp_cond, axis=1) - np.log(n_z_samples)
        log_num_avg = logsumexp(log_num) - np.log(N)
        log_den_avg = logsumexp(log_den) - np.log(N)

        return np.exp(log_num_avg - log_den_avg)


    def posterior_predictive(self, cond_obs, n_z_samples=50):
        """
        Predict P(Y_last=1 | Y_0:S-2 = cond_obs, data)
        using Monte Carlo over latent Z samples (non-vectorized).
    
        cond_obs: array of shape (S-1,), binary assignment of observed features
        n_z_samples: number of latent draws per posterior sample
        """
        cond_obs = np.array(cond_obs)        # (S-1,)
        N, K, S = self.good_samples_W.shape
    
        assert len(cond_obs) == S-1, "cond_obs must have length S-1"
    
        log_num_list = []
        log_den_list = []
    
        for n in range(N):  # loop over posterior samples
            W = self.good_samples_W[n]      # (K, S)
            b = self.good_samples_b[n]      # (S,)
            Z_post = self.good_samples_Z[n] # (T, K)
    
            # Compute empirical prior for Z
            Z_counts = Z_post.sum(axis=0)   # (K,)
            prior_prob = Z_counts / self.T
    
            # Monte Carlo over latent draws
            for mc in range(n_z_samples):
                Z_new = np.random.binomial(1, prior_prob)  # (K,)
    
                # ----- Full observation log-prob (joint) -----
                eta_full = Z_new @ W + b               # (S,)
                obs_full = np.concatenate([cond_obs, [1]])  # full obs vector
                logp_full = np.sum(
                    obs_full * np.log(expit(eta_full) + 1e-12) +
                    (1 - obs_full) * np.log(1 - expit(eta_full) + 1e-12)
                )
                log_num_list.append(logp_full)
    
                # ----- Conditioned dimensions only -----
                eta_cond = eta_full[:S-1]
                logp_cond = np.sum(
                    cond_obs * np.log(expit(eta_cond) + 1e-12) +
                    (1 - cond_obs) * np.log(1 - expit(eta_cond) + 1e-12)
                )
                log_den_list.append(logp_cond)
    
        # Convert lists to arrays for logsumexp
        log_num_arr = np.array(log_num_list).reshape(N, n_z_samples)
        log_den_arr = np.array(log_den_list).reshape(N, n_z_samples)
    
        # Monte Carlo estimate in log-space
        log_num = logsumexp(log_num_arr, axis=1) - np.log(n_z_samples)  # average over latent draws
        log_den = logsumexp(log_den_arr, axis=1) - np.log(n_z_samples)
    
        # Average over posterior samples
        log_num_avg = logsumexp(log_num)
        log_den_avg = logsumexp(log_den)
        print(f'log_numerator: {log_num_avg}')
        print(f'log_denom: {log_den_avg}')
    
        # Return conditional probability
        p_post = np.exp(log_num_avg - log_den_avg)
        return p_post