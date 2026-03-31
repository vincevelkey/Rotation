from __future__ import annotations

import numpy as np


STIMULUS_NAMES = ("A", "X", "B", "US")


def generate_second_order_conditioning(
    ax_trials: int = 4,
    a_us_trials: int = 96,
    b_us_trials: int = 8,
    unpaired_x_trials: int = 0,
    shuffle: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Construct the four-stimulus protocol used in Courville et al. (2003).

    Columns are ordered as (A, X, B, US).
    """
    rows = []
    rows.extend([[1, 0, 0, 1]] * a_us_trials)
    rows.extend([[0, 0, 1, 1]] * b_us_trials)
    rows.extend([[1, 1, 0, 0]] * ax_trials)
    rows.extend([[0, 1, 0, 0]] * unpaired_x_trials)
    data = np.asarray(rows, dtype=np.int8)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(data, axis=0)
    return data


def conditioning_query(
    present: tuple[str, ...] | list[str],
    stimulus_names: tuple[str, ...] = STIMULUS_NAMES,
    unobserved: tuple[str, ...] | list[str] = ("US",),
    condition_absent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed values and a mask for a conditioning query.

    By default this matches the paper's test notation:
    queried conditioned stimuli are set to 1, the remaining non-US stimuli are
    conditioned to be absent, and US is left unobserved.
    """
    present_set = set(present)
    unobserved_set = set(unobserved)
    values = np.zeros(len(stimulus_names), dtype=np.int8)
    mask = np.zeros(len(stimulus_names), dtype=bool)
    for idx, name in enumerate(stimulus_names):
        if condition_absent and name not in unobserved_set:
            mask[idx] = True
        if name in present_set:
            values[idx] = 1
            mask[idx] = True
        if name in unobserved_set:
            mask[idx] = False
    return values, mask
