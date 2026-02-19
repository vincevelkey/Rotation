import numpy as np
import matplotlib.pyplot as plt
from models import ParametricLLFM
from models import GibbsSamplerLLFM

def latent_features(threshold=5, Z_post=None, W_post=None, b_post=None):

    threshold = 5

    # Normal dictionary
    groups = {}
    zeros = 0
    n_samples, T, K = Z_post.shape
    S = W_post.shape[2]
    avg_active = 0
    for it in range(n_samples):
        Z = Z_post[it]   # (T, K)
        W = W_post[it]   # (K, S)
        b = b_post[it]

        usage = Z.sum(axis=0)

        # Identify active features
        active_idx = np.where(usage > threshold)[0]
        k_active = len(active_idx)
        avg_active += k_active
        if k_active == 0:

            zeros += 1
            continue  # Skip samples with no active features

        # Sort active features by usage descending
        sorted_idx = active_idx[np.argsort(usage[active_idx])[::-1]]

        W_perm = W[sorted_idx, :]

        # ---- Manually create group if needed ----
        if k_active not in groups:
            groups[k_active] = []

        groups[k_active].append((W_perm, b))
    avg_active /= n_samples

    # ---- Report results ----
    print("Posterior grouping by number of active features\n")
    print(f"Number of samples with zero active features: {zeros}/{n_samples}\n")
    for k in sorted(groups.keys()):
        samples = groups[k]
        n_group = len(samples)

        print(f"Group with {k} active features:")
        print(f"  Number of posterior samples: {n_group}/{n_samples}")

        W_stack = np.array([w for w, _ in samples])   # (n_group, k, S)
        b_stack = np.array([b for _, b in samples])   # (n_group, S)

        W_mean = W_stack.mean(axis=0)
        b_mean = b_stack.mean(axis=0)

        print("  Average weights:")
        print(W_mean)

        print("  Average bias:")
        print(b_mean)
        print("-" * 40)
    
    return avg_active


def latent_features_to_file(filename,
                            threshold=5,
                            Z_post=None,
                            W_post=None,
                            b_post=None,
                            header=None):

    groups = {}
    zeros = 0
    n_samples, T, K = Z_post.shape
    S = W_post.shape[2]
    avg_active = 0

    for it in range(n_samples):
        Z = Z_post[it]
        W = W_post[it]
        b = b_post[it]

        usage = Z.sum(axis=0)
        active_idx = np.where(usage > threshold)[0]
        k_active = len(active_idx)

        avg_active += k_active

        if k_active == 0:
            zeros += 1
            continue

        sorted_idx = active_idx[np.argsort(usage[active_idx])[::-1]]
        W_perm = W[sorted_idx, :]

        if k_active not in groups:
            groups[k_active] = []

        groups[k_active].append((W_perm, b))

    avg_active /= n_samples

    # ---- Write to file ----
    with open(filename, "a") as f:

        f.write("\n" + "="*60 + "\n")

        if header is not None:
            f.write(f"{header}\n")
            f.write("-"*60 + "\n")

        f.write("Posterior grouping by number of active features\n\n")
        f.write(f"Number of samples with zero active features: {zeros}/{n_samples}\n")
        f.write(f"Average active features per sample: {avg_active:.3f}\n\n")

        for k in sorted(groups.keys()):
            samples = groups[k]
            n_group = len(samples)

            f.write(f"Group with {k} active features:\n")
            f.write(f"  Number of posterior samples: {n_group}/{n_samples}\n")

            W_stack = np.array([w for w, _ in samples])
            b_stack = np.array([b for _, b in samples])

            W_mean = W_stack.mean(axis=0)
            b_mean = b_stack.mean(axis=0)

            f.write("  Average weights:\n")
            f.write(np.array2string(W_mean, precision=3))
            f.write("\n")

            f.write("  Average bias:\n")
            f.write(np.array2string(b_mean, precision=3))
            f.write("\n")
            f.write("-"*40 + "\n")

    return avg_active


def latent_features_dominant_to_file(filename,
                                     threshold=5,
                                     Z_post=None,
                                     W_post=None,
                                     b_post=None,
                                     header=None):

    groups = {}
    n_samples, T, K = Z_post.shape

    for it in range(n_samples):
        Z = Z_post[it]
        W = W_post[it]
        b = b_post[it]

        usage = Z.sum(axis=0)
        active_idx = np.where(usage > threshold)[0]
        k_active = len(active_idx)

        if k_active == 0:
            continue

        sorted_idx = active_idx[np.argsort(usage[active_idx])[::-1]]
        W_perm = W[sorted_idx, :]

        if k_active not in groups:
            groups[k_active] = []

        groups[k_active].append((W_perm, b))

    if len(groups) == 0:
        return 0

    # Find group with most samples
    dominant_k = max(groups.keys(), key=lambda k: len(groups[k]))
    samples = groups[dominant_k]
    n_group = len(samples)

    W_stack = np.array([w for w, _ in samples])
    b_stack = np.array([b for _, b in samples])

    W_mean = W_stack.mean(axis=0)
    b_mean = b_stack.mean(axis=0)

    # ---- Write to file ----
    with open(filename, "a") as f:

        f.write("\n" + "="*60 + "\n")

        if header is not None:
            f.write(f"{header}\n")
            f.write("-"*60 + "\n")

        f.write("Dominant posterior feature count summary\n\n")
        f.write(f"Most frequent number of active features: {dominant_k}\n")
        f.write(f"Number of posterior samples in this group: {n_group}/{n_samples}\n\n")

        f.write("Average weights (dominant group):\n")
        f.write(np.array2string(W_mean, precision=3))
        f.write("\n")

        f.write("Average bias (dominant group):\n")
        f.write(np.array2string(b_mean, precision=3))
        f.write("\n")

    return dominant_k
