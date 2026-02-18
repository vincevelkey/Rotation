import numpy as np
from scipy.special import expit  # sigmoid
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

        self.samples_W = []
        self.samples_b = []
        self.samples_Z = []
        self.feature_counter = 0

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

            self.samples_W.append(self.model.W.copy())
            self.samples_b.append(self.model.b.copy())
            self.samples_Z.append(self.model.Z.copy())

            self.feature_counter += (self.model.Z.sum(axis=0) > 0).sum()
        self.feature_counter /= self.n_iter  # Average number of active features across iterations
        print(f"Average number of active features across iterations: {self.feature_counter:.2f}")

    def get_posterior_samples(self):
        burn = self.burn
        n_subsample = self.n_subsample

        total = len(self.samples_W)
        valid_idx = np.arange(burn, total)

        if n_subsample is None or n_subsample >= len(valid_idx):
            chosen = valid_idx
        else:
            chosen = np.random.choice(valid_idx, size=n_subsample, replace=False)

        samples_W = [self.samples_W[i] for i in chosen]
        samples_b = [self.samples_b[i] for i in chosen]
        samples_Z = [self.samples_Z[i] for i in chosen]
        return samples_W, samples_b, samples_Z



    def posterior_predictive(self, burn=0, nsamples=None,
                         conds=[1,0,0], predindex=3,
                         n_cond_gibbs=5):

        samples_W, samples_b, samples_Z = self.get_posterior_samples(
            burn=burn, n_subsample=nsamples
        )

        probs = []
        S_cond = len(conds)

        for it in range(len(samples_W)):

            W = samples_W[it]
            b = samples_b[it]
            Z = samples_Z[it]

            # ---- Predictive prior for Z_new ----
            m_k = Z.sum(axis=0)
            prior_prob = (m_k + self.alpha / self.K) / (
                self.T + self.alpha / self.K + 1
            )

            # ---- Initialize Z_new ----
            Z_new = np.random.binomial(1, prior_prob)

            # ---- Compute eta for conditioned dims ONCE ----
            W_cond = W[:, :S_cond]
            b_cond = b[:S_cond]

            eta = Z_new @ W_cond + b_cond  # shape (S_cond,)

            # ---- Gibbs sweeps ----
            for sweep in range(n_cond_gibbs):

                for k in range(self.K):

                    z_old = Z_new[k]

                    # Remove old contribution if active
                    if z_old == 1:
                        eta_minus = eta - W_cond[k]
                    else:
                        eta_minus = eta

                    # ----- z = 0 -----
                    ll0 = np.sum(
                        np.array(conds) * np.log(expit(eta_minus) + 1e-12) +
                        (1 - np.array(conds)) * np.log(1 - expit(eta_minus) + 1e-12)
                    )

                    # ----- z = 1 -----
                    eta_plus = eta_minus + W_cond[k]
                    ll1 = np.sum(
                        np.array(conds) * np.log(expit(eta_plus) + 1e-12) +
                        (1 - np.array(conds)) * np.log(1 - expit(eta_plus) + 1e-12)
                    )

                    # Prior
                    log_prior_1 = np.log(prior_prob[k] + 1e-12)
                    log_prior_0 = np.log(1 - prior_prob[k] + 1e-12)

                    logp1 = ll1 + log_prior_1
                    logp0 = ll0 + log_prior_0

                    p = 1.0 / (1.0 + np.exp(-(logp1 - logp0)))

                    z_new = np.random.binomial(1, p)
                    Z_new[k] = z_new

                    # Update eta incrementally
                    if z_new != z_old:
                        eta = eta_minus + z_new * W_cond[k]

            # ---- Predictive probability ----
            eta_pred = Z_new @ W[:, predindex] + b[predindex]
            p_pred = expit(eta_pred)

            probs.append(p_pred)

        return np.mean(probs)



