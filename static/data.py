import numpy as np

from .models import ParametricLLFM


def generate_data_second_order(axnum=1):
    """
    Generate the second-order conditioning dataset from Courville et al. (2003).
    Observations are encoded as (A, X, B, US).
    The variable axnum gives the number of A-X pairings without reinforcement.
    """

    D = np.zeros((104 + axnum, 4))
    D[:96, :] = [1, 0, 0, 1]
    D[96:104, :] = [0, 0, 1, 1]
    D[104:, :] = [1, 1, 0, 0]

    return D


def generate_parametric(T=500, S=4, K=2, alpha=1.0, rho=0.1, sigma_w=3.0, mu_b=-1.0, sigma_b=1.0):

    model = ParametricLLFM(T=T, S=S, K=K, alpha=alpha, rho=rho, sigma_w=sigma_w, mu_b=mu_b, sigma_b=sigma_b)
    Y = model.sample_observations()
    W_effective = model.W * model.A

    return Y, model.Z, W_effective, model.b, model.pi


def generate_data_li_ov(trials=50, latent_inhibition=True, overshadowing=True):
    """
    Generate the LI/OV dataset with a 50-trial zero baseline, an optional
    50-trial latent-inhibition block, and a final reinforced block.
    """
    if trials <= 0:
        trials = 50

    n_rows = 50 + trials + (50 if latent_inhibition else 0)
    D = np.zeros((n_rows, 3))
    next_row = 50

    if latent_inhibition:
        D[next_row:next_row + 50, :] = [1, 0, 0]
        next_row += 50

    if overshadowing:
        D[next_row:, :] = [1, 1, 1]
    else:
        D[next_row:, :] = [1, 0, 1]

    return D
