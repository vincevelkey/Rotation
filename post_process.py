import numpy as np
from models import ParametricLLFM
from models import GibbsSamplerLLFM
from scipy.optimize import linear_sum_assignment


def frequency_align(Z_post=None, W_post=None, b_post=None, W_true=None, p_true=None):

    n_samples, T, K = Z_post.shape
    S = W_post.shape[2]

    for it in range(n_samples):
        Z = Z_post[it]   # (T, K)
        W = W_post[it]   # (K, S)
        b = b_post[it]
        usage = Z.sum(axis=0)  # (K,)

        sorted_idx = np.argsort(usage)[::-1]
        Z_post[it] = Z[:, sorted_idx]
        W_post[it] = W[sorted_idx, :]

    sorted_idx_true = np.argsort(p_true)[::-1]
    W_true_sorted = W_true[sorted_idx_true, :]  
    p_true_sorted = p_true[sorted_idx_true]

    return Z_post, W_post, b_post, W_true_sorted, p_true_sorted

def hungarian_align(Z_post=None, W_post=None, b_post=None, W_true=None):

    n_samples, T, K = Z_post.shape
    S = W_post.shape[2]

    for it in range(n_samples):
        Z = Z_post[it]   # (T, K)
        W = W_post[it]   # (K, S)
        b = b_post[it]

        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.linalg.norm(W_true[i] - W[j])

        row_ind, col_ind = linear_sum_assignment(cost)

        Z_post[it] = Z[:, col_ind]
        W_post[it] = W[col_ind, :]
        # b unaffected by permutation

    return Z_post, W_post, b_post