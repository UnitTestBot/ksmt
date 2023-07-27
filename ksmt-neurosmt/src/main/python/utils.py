import numpy as np


def train_val_test_indices(cnt, val_pct=0.15, test_pct=0.1):
    perm = np.arange(cnt)
    np.random.shuffle(perm)

    val_cnt = int(cnt * val_pct)
    test_cnt = int(cnt * test_pct)

    return perm[val_cnt + test_cnt:], perm[:val_cnt], perm[val_cnt:val_cnt + test_cnt]


def align_sat_unsat_sizes(sat_data, unsat_data):
    sat_indices = list(range(len(sat_data)))
    unsat_indices = list(range(len(unsat_data)))

    if len(sat_indices) > len(unsat_indices):
        sat_indices = np.random.choice(np.array(sat_indices), len(unsat_indices), replace=False)
    elif len(sat_indices) < len(unsat_indices):
        unsat_indices = np.random.choice(np.array(unsat_indices), len(sat_indices), replace=False)

    return (
        list(np.array(sat_data, dtype=object)[sat_indices]),
        list(np.array(unsat_data, dtype=object)[unsat_indices])
    )
