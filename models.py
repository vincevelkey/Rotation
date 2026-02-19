import numpy as np
from scipy.special import expit, logsumexp # sigmoid
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
                 sigma_w=3, sigma_b=1.0):

        self.T = T
        self.S = S
        self.K = K
        self.alpha = alpha
        self.mu_b = mu_b
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        # -------------------------------------------------
        # Beta-Bernoulli prior for features
        # -------------------------------------------------

        self.pi = np.random.beta(alpha / K, 1.0, size=K)
        self.Z = np.random.binomial(1, self.pi, size=(T, K))
        self.W = np.random.normal(0, sigma_w, size=(K, S))
        self.b = np.random.normal(0, sigma_b, size=(S,))

    # -----------------------------------------------------

    def linear_predictor(self):
        return self.Z @ self.W + self.b # (T × S)

    def sigmoid(self):
        return expit(self.linear_predictor())

    def sample_observations(self):
        P = self.sigmoid()
        return np.random.binomial(1, P) # (T × S)


# ============================================================
# Gibbs Sampler with Polya-Gamma augmentation
# ============================================================

class GibbsSamplerLLFM:
    """
    Collapsed Gibbs sampler for finite-K logistic latent feature model
    with Polya–Gamma augmentation and Beta-Bernoulli prior
    (π integrated out).
    """

    def __init__(self, Data, K=15, alpha=0.5, sigma_w=3.0, sigma_b=1.0, mu_b=-15.0, n_iter=1000, burn=200, n_subsample=None):

        self.Data = Data
        self.T, self.S = Data.shape

        self.K = K  # Number of latent features (can be tuned)
        self.alpha = alpha  # IBP concentration parameter
        self.sigma_w = sigma_w  # Prior std for weights
        self.sigma_b = sigma_b  # Prior std for bias   
        self.mu_b = mu_b  # Prior mean for bias
        self.model = ParametricLLFM(self.T, self.S, self.K, self.alpha,
                                    self.mu_b, self.sigma_w, self.sigma_b) 

        self.n_iter = n_iter
        self.burn = burn
        self.n_subsample = n_subsample

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

            # -------------------------------------------------
            # 1) Sample Polya-Gamma variables
            # -------------------------------------------------

            eta = self.model.linear_predictor()
            omega = random_polyagamma(1, eta)

            # -------------------------------------------------
            # 2) Sample W
            # -------------------------------------------------

            for s in range(self.S):

                Z = self.model.Z
                omega_d = omega[:, s]
                kappa = self.Data[:, s] - 0.5

                Omega = np.diag(omega_d)

                V_inv = Z.T @ Omega @ Z + np.eye(self.K) / self.sigma_w**2
                rhs = Z.T @ kappa

                L = np.linalg.cholesky(V_inv)

                temp = np.linalg.solve(L, rhs)
                mu = np.linalg.solve(L.T, temp)

                noise = np.linalg.solve(L.T, np.random.randn(self.K))
                self.model.W[:, s] = mu + noise
            


            # -------------------------------------------------
            # 3) Sample bias b with non-zero prior mean mu_b
            # -------------------------------------------------

            for s in range(self.S):

                omega_d = omega[:, s]                 # Polya-Gamma variables
                kappa = self.Data[:, s] - 0.5         # y - 1/2
                linear_part = self.model.Z @ self.model.W[:, s]

                # Posterior variance
                V_b = 1.0 / (omega_d.sum() + 1.0 / self.sigma_b**2)

                # Posterior mean (includes prior mean term!)
                m_b = V_b * (
                    np.sum(kappa - omega_d * linear_part)
                    + self.mu_b / self.sigma_b**2
                )

                # Gibbs draw
                self.model.b[s] = m_b + np.sqrt(V_b) * np.random.randn()


            # -------------------------------------------------
            # 4) Collapsed Z update (π integrated out)
            # -------------------------------------------------
            for t in range(self.T):

                # Compute eta_t once
                eta_t = self.model.Z[t] @ self.model.W + self.model.b

                for k in range(self.K):

                    z_old = self.model.Z[t, k]

                    # Compute m_{-t,k}
                    m_k = self.model.Z[:, k].sum() - z_old

                    # Collapsed prior
                    log_prior_1 = np.log(m_k + self.alpha / self.K + 1e-12)
                    log_prior_0 = np.log(self.T - m_k + 1e-12)

                    # Remove current contribution if z_old = 1
                    if z_old == 1:
                        eta_minus = eta_t - self.model.W[k]
                    else:
                        eta_minus = eta_t

                    # ----- z = 0 -----
                    ll0 = np.sum(
                        self.Data[t] * np.log(expit(eta_minus) + 1e-12) +
                        (1 - self.Data[t]) * np.log(1 - expit(eta_minus) + 1e-12)
                    )

                    # ----- z = 1 -----
                    eta_plus = eta_minus + self.model.W[k]
                    ll1 = np.sum(
                        self.Data[t] * np.log(expit(eta_plus) + 1e-12) +
                        (1 - self.Data[t]) * np.log(1 - expit(eta_plus) + 1e-12)
                    )

                    # Posterior
                    logp0 = ll0 + log_prior_0
                    logp1 = ll1 + log_prior_1

                    p1 = 1.0 / (1.0 + np.exp(-(logp1 - logp0)))

                    z_new = np.random.binomial(1, p1)

                    self.model.Z[t, k] = z_new

                    # Update eta_t incrementally
                    if z_new != z_old:
                        eta_t = eta_minus + z_new * self.model.W[k]


            # -------------------------------------------------
            # Store samples
            # -------------------------------------------------

            self.samples_W[it] = self.model.W
            self.samples_b[it] = self.model.b
            self.samples_Z[it] = self.model.Z


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
    

    def posterior_predictive(self, cond_obs, n_z_samples=50):
        """
        Predict P(Y_last | Y_0:S-2 = cond_obs, data)
        cond_obs: array of shape (S-1,), binary assignment of observed features
        """
        samples_W = self.good_samples_W   # (N, K, S)
        samples_b = self.good_samples_b   # (N, S)
        samples_Z = self.good_samples_Z   # (N, T, K)
        cond_obs = np.array(cond_obs)     # (S-1,)
        N, K, S = samples_W.shape

        # Check cond_obs length
        assert len(cond_obs) == S-1, "cond_obs must have length S-1"

        # ----- Repeat latent draws -----
        Z_train_counts = samples_Z.sum(axis=1)  # (N, K)
        prior_prob = Z_train_counts / self.T
        Z_new = np.random.binomial(1, prior_prob[:, None, :], size=(N, n_z_samples, K))  # (N, n_z_samples, K)

        # ----- Expand W and b -----
        W_expand = samples_W[:, None, :, :]        # (N,1,K,S)
        W_expand = np.repeat(W_expand, n_z_samples, axis=1)
        b_expand = samples_b[:, None, :]           # (N,1,S)
        b_expand = np.repeat(b_expand, n_z_samples, axis=1)

        # ----- Build full observation vector: conditioned + prediction -----
        obs = np.concatenate([cond_obs, [1]])      # predict "1" for last index
        indices = list(range(S))                    # all dimensions

        # ----- Compute eta for all relevant dims -----
        eta_all = np.einsum('nzk,nzks->nzs', Z_new, W_expand[:, :, :, indices]) + b_expand[:, :, indices]  # (N, n_z_samples, S)

        obs_expand = obs[None, None, :]  # (1,1,S)
        logp_all = obs_expand * np.log(expit(eta_all)+1e-12) + (1-obs_expand) * np.log(1-expit(eta_all)+1e-12)
        logp_all = np.sum(logp_all, axis=2)  # sum over dimensions -> (N, n_z_samples)

        # ----- Marginal for conditioned dims only -----
        eta_cond = eta_all[:, :, :S-1]
        obs_cond = cond_obs[None, None, :]     # (1,1,S-1)
        logp_cond = obs_cond * np.log(expit(eta_cond)+1e-12) + (1-obs_cond) * np.log(1-expit(eta_cond)+1e-12)
        logp_cond = np.sum(logp_cond, axis=2)  # (N, n_z_samples)

        # ----- Monte Carlo estimate -----
        log_num = logsumexp(logp_all, axis=1) - np.log(n_z_samples)
        log_den = logsumexp(logp_cond, axis=1) - np.log(n_z_samples)

        log_num_avg = logsumexp(log_num) - np.log(N)
        log_den_avg = logsumexp(log_den) - np.log(N)

        p_post = np.exp(log_num_avg - log_den_avg)
        return p_post



    