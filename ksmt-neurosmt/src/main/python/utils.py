import numpy as np


def train_val_test_indices(n, val_pct=0.15, test_pct=0.1):
    perm = np.arange(n)
    np.random.shuffle(perm)

    val_cnt = int(n * val_pct)
    test_cnt = int(n * test_pct)

    return perm[val_cnt + test_cnt:], perm[:val_cnt], perm[val_cnt:val_cnt + test_cnt]
