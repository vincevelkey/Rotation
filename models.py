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


# ============================================================
# Gibbs Sampler with Polya-Gamma augmentation
# ============================================================

class GibbsSamplerLLFM:
    """
    Collapsed Gibbs sampler for finite-K logistic latent feature model
    with Polya–Gamma augmentation and Beta-Bernoulli prior
    (π integrated out). Optionally allows fixed bias.
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
        self.model = ParametricLLFM(T=self.T, S=self.S, K=self.K, rho=self.rho, alpha=self.alpha,
                                    mu_b=self.mu_b, sigma_w=self.sigma_w, sigma_b=self.sigma_b)

        # Gibbs sampling parameters
        self.n_iter = n_iter
        self.burn = burn
        self.n_subsample = n_subsample

        # Storage
        self.samples_W = np.empty((self.n_iter, self.K, self.S))
        self.samples_b = np.empty((self.n_iter, self.S))
        self.samples_Z = np.empty((self.n_iter, self.T, self.K))
        self.samples_A = np.empty((self.n_iter, self.K, self.S))
        self.good_samples_W = np.empty((self.n_subsample, self.K, self.S))
        self.good_samples_b = np.empty((self.n_subsample, self.S))
        self.good_samples_Z = np.empty((self.n_subsample, self.T, self.K))
        self.good_samples_A = np.empty((self.n_subsample, self.K, self.S))

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
                omega_s = omega[:, s]
                Z_mask = Z * a
                kappa = self.Data[:, s] - 0.5
                ZOmega = omega_s[:, None] * Z_mask
                V_inv = Z_mask.T @ ZOmega + np.eye(self.K) / self.sigma_w**2
                b_s = self.model.b[s]
                rhs = Z_mask.T @ (kappa - omega_s * b_s)
                L = np.linalg.cholesky(V_inv)
                mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
                noise = np.linalg.solve(L.T, np.random.randn(self.K))
                w_new = mu + noise
                w_new *= a  # zero out entries where a=0
                self.model.W[:, s] = w_new

            # 3) Sample b
            for s in range(self.S):
                omega_s = omega[:, s]
                kappa = self.Data[:, s] - 0.5
                linear_part = self.model.Z @ (self.model.A[:, s] * self.model.W[:, s])
                V_b = 1.0 / (omega_s.sum() + 1.0 / self.sigma_b**2)
                m_b = V_b * (np.sum(kappa - omega_s * linear_part) + self.mu_b / self.sigma_b**2)
                self.model.b[s] = m_b[s] + np.sqrt(V_b) * np.random.randn()

            # 4) Sample A

            for k in range(self.K):
                for s in range(self.S):

                    w = self.model.W[k, s]
                    z = self.model.Z[:, k]

                    eta = self.model.linear_predictor()

                    # remove contribution
                    eta_minus = eta[:, s] - z * w

                    # likelihood if A=0
                    p0 = expit(eta_minus)

                    ll0 = np.sum(
                        self.Data[:, s] * np.log(p0 + 1e-12) +
                        (1 - self.Data[:, s]) * np.log(1 - p0 + 1e-12)
                    )

                    # likelihood if A=1
                    eta_plus = eta_minus + z * w
                    p1 = expit(eta_plus)

                    ll1 = np.sum(
                        self.Data[:, s] * np.log(p1 + 1e-12) +
                        (1 - self.Data[:, s]) * np.log(1 - p1 + 1e-12)
                    )

                    logp1 = ll1 + np.log(self.rho + 1e-12)
                    logp0 = ll0 + np.log(1 - self.rho + 1e-12)

                    prob = 1 / (1 + np.exp(-(logp1 - logp0)))

                    a_new = np.random.binomial(1, prob)

                    self.model.A[k, s] = a_new

                    if a_new == 0:
                        self.model.W[k, s] = 0.0

            # 4) Collapsed Z update
            for t in range(self.T):

                eta_t = self.model.Z[t] @ (self.model.A * self.model.W) + self.model.b

                for k in range(self.K):

                    z_old = self.model.Z[t, k]

                    m_k = self.model.Z[:, k].sum() - z_old

                    log_prior1 = np.log(m_k + self.alpha / self.K + 1e-12)
                    log_prior0 = np.log(self.T - m_k + 1e-12)

                    contrib = self.model.A[k] * self.model.W[k]

                    eta_minus = eta_t - contrib if z_old else eta_t

                    # z=0
                    p0 = expit(eta_minus)

                    ll0 = np.sum(
                        self.Data[t] * np.log(p0 + 1e-12) +
                        (1 - self.Data[t]) * np.log(1 - p0 + 1e-12)
                    )

                    # z=1
                    eta_plus = eta_minus + contrib
                    p1 = expit(eta_plus)

                    ll1 = np.sum(
                        self.Data[t] * np.log(p1 + 1e-12) +
                        (1 - self.Data[t]) * np.log(1 - p1 + 1e-12)
                    )

                    logp0 = ll0 + log_prior0
                    logp1 = ll1 + log_prior1

                    prob = 1 / (1 + np.exp(-(logp1 - logp0)))

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
        #return self.good_samples_W, self.good_samples_b, self.good_samples_Z



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
    
                # ----- Conditioned dimensions only -----
                eta_cond = eta_full[:S-1]
                logp_cond = np.sum(
                    cond_obs * np.log(expit(eta_cond) + 1e-12) +
                    (1 - cond_obs) * np.log(1 - expit(eta_cond) + 1e-12)
                )
                log_pz = np.sum(
                 Z_new * np.log(prior_prob + 1e-12) +
                (1 - Z_new) * np.log(1 - prior_prob + 1e-12)
                )

                log_num_list.append(logp_full + log_pz)
                log_den_list.append(logp_cond + log_pz)
    
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
    

    def posterior_predictive_gibbs(self, cond_obs, n_z_samples=50, gibbs_steps=5):

        cond_obs = np.array(cond_obs)

        N, K, S = self.good_samples_W.shape
        probs = []

        for n in range(N):

            W = self.good_samples_W[n]
            A = self.good_samples_A[n]
            b = self.good_samples_b[n]
            Z_post = self.good_samples_Z[n]

            W_eff = A * W

            # empirical prior
            pi = Z_post.mean(axis=0)

            for _ in range(n_z_samples):

                z = np.random.binomial(1, pi)

                # compute predictor once
                eta = z @ W_eff + b

                # Gibbs sampling for z | y_cond
                for _ in range(gibbs_steps):

                    for k in range(K):

                        z_old = z[k]
                        w = W_eff[k]

                        # remove current contribution
                        eta_minus = eta - z_old * w

                        # z = 0
                        p0 = expit(eta_minus[:S-1])
                        ll0 = np.sum(
                            cond_obs*np.log(p0+1e-12) +
                            (1-cond_obs)*np.log(1-p0+1e-12)
                        ) + np.log(1-pi[k]+1e-12)

                        # z = 1
                        eta_plus = eta_minus + w
                        p1 = expit(eta_plus[:S-1])
                        ll1 = np.sum(
                            cond_obs*np.log(p1+1e-12) +
                            (1-cond_obs)*np.log(1-p1+1e-12)
                        ) + np.log(pi[k]+1e-12)

                        p = 1/(1+np.exp(-(ll1-ll0)))

                        z_new = np.random.binomial(1, p)

                        z[k] = z_new

                        # update eta incrementally
                        eta = eta_minus + z_new * w

                probs.append(expit(eta[-1]))

        return np.mean(probs)