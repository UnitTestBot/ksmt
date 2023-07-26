import numpy as np


def train_val_test_indices(n, val_pct=0.15, test_pct=0.1):
    perm = np.arange(n)
    np.random.shuffle(perm)

    val_cnt = int(n * val_pct)
    test_cnt = int(n * test_pct)

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

    """
    if len(sat_data) > len(unsat_data):
        return list(np.random.choice(np.array(sat_data, dtype=object), len(unsat_data), replace=False)), unsat_data
    elif len(sat_data) < len(unsat_data):
        print(type(unsat_data[0]))
        print(type(np.array(unsat_data, dtype=np.object)[0]))
        return sat_data, list(np.random.choice(np.array(unsat_data, dtype=object), len(sat_data), replace=False))
    else:
        return sat_data, unsat_data
    """
