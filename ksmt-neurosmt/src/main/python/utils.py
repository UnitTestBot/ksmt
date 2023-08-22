import os

import numpy as np
from tqdm import tqdm

from GraphReader import read_graph_by_path
from GraphDataloader import MAX_FORMULA_SIZE, MAX_FORMULA_DEPTH


def train_val_test_indices(cnt, val_qty=0.15, test_qty=0.1):
    perm = np.arange(cnt)
    np.random.shuffle(perm)

    val_cnt = int(cnt * val_qty)
    test_cnt = int(cnt * test_qty)

    return perm[val_cnt + test_cnt:], perm[:val_cnt], perm[val_cnt:val_cnt + test_cnt]


def select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, paths):
    correct_paths = []
    for path in tqdm(paths):
        operators, edges, _ = read_graph_by_path(path, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH)

        if operators is None:
            continue

        if len(edges) == 0:
            print(f"w: ignoring formula without edges; file '{path}'")
            continue

        correct_paths.append(os.path.relpath(path, path_to_dataset_root))

    return correct_paths


def align_sat_unsat_sizes_with_upsamping(sat_data, unsat_data):
    sat_cnt = len(sat_data)
    unsat_cnt = len(unsat_data)

    sat_indices = list(range(sat_cnt))
    unsat_indices = list(range(unsat_cnt))

    if sat_cnt < unsat_cnt:
        sat_indices += list(np.random.choice(np.array(sat_indices), unsat_cnt - sat_cnt, replace=True))
    elif sat_cnt > unsat_cnt:
        unsat_indices += list(np.random.choice(np.array(unsat_indices), sat_cnt - unsat_cnt, replace=False))

    return (
        list(np.array(sat_data, dtype=object)[sat_indices]),
        list(np.array(unsat_data, dtype=object)[unsat_indices])
    )


def align_sat_unsat_sizes_with_downsamping(sat_data, unsat_data):
    sat_cnt = len(sat_data)
    unsat_cnt = len(unsat_data)

    sat_indices = list(range(sat_cnt))
    unsat_indices = list(range(unsat_cnt))

    if sat_cnt > unsat_cnt:
        sat_indices = np.random.choice(np.array(sat_indices), unsat_cnt, replace=False)
    elif sat_cnt < unsat_cnt:
        unsat_indices = np.random.choice(np.array(unsat_indices), sat_cnt, replace=False)

    return (
        list(np.array(sat_data, dtype=object)[sat_indices]),
        list(np.array(unsat_data, dtype=object)[unsat_indices])
    )


def align_sat_unsat_sizes(sat_data, unsat_data, mode):
    if mode == "none":
        return sat_data, unsat_data
    elif mode == "upsample":
        return align_sat_unsat_sizes_with_upsamping(sat_data, unsat_data)
    elif mode == "downsample":
        return align_sat_unsat_sizes_with_downsamping(sat_data, unsat_data)
    else:
        raise Exception(f"unknown sampling mode {mode}")
