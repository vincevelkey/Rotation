import numpy as np
from models import ParametricLLFM
from scipy.special import expit

def generate_data_second_order(axnum=1):
    '''
    Generate data for second-order conditioning task. The variable axnum gives the number of X-A pairings.
    '''

    D = np.zeros((210 + axnum, 4))
    D[:100, :] = [0,0,0,0]
    D[100:200, :] = [1,0,0,1]
    D[200:210, :] = [0, 0, 1,1]
    D[210:, :] = [1, 1, 0, 0]  # Additional rows for axnum > 0

    return D


def generate_synthetic(T=150, S=4, K_true=2, alpha=1.0, bias=-6.0, weights=6.0):

    # ----- Generate feature probabilities -----
    p_true = np.random.beta(alpha / K_true, 1.0, size=K_true)

    # ----- True latent features -----
    Z_true = np.random.binomial(1, p_true, size=(T, K_true))
    

    # ----- True weights -----
    W_true = np.zeros((K_true, S))
    W_true[0,1] = weights
    W_true[0,3] = weights
    W_true[1,0] = weights
    W_true[1,2] = weights


    # ----- True bias -----
    b_true = np.full(S, bias)  # Bias for each stimulus

    # ----- Generate observations -----
    logits = Z_true @ W_true + b_true
    P_true = expit(logits)
    Y = np.random.binomial(1, P_true)

    return Y, Z_true, W_true, b_true, p_true


def generate_parametric(T=500, S=4, K=2, alpha=1.0, rho=0.1, sigma_w=3.0, mu_b=-1.0, sigma_b=1.0):

    model = ParametricLLFM(T=T, S=S, K=K, alpha=alpha, rho=rho, sigma_w=sigma_w, mu_b=mu_b, sigma_b=sigma_b)
    Y = model.sample_observations()
    W_effective = model.W * model.A

    return Y, model.Z, W_effective, model.b, model.pi
