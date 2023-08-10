#!/usr/bin/python3

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import trange

from pytorch_lightning import seed_everything

from GraphDataloader import METADATA_PATH
from utils import train_val_test_indices, align_sat_unsat_sizes, select_paths_with_suitable_samples_and_transform_to_paths_from_root


SHRINK = 10 ** (int(os.environ["SHRINK"]) if "SHRINK" in os.environ else 20)


def classic_random_split(path_to_dataset_root, val_qty, test_qty, align_train_mode, align_val_mode, align_test_mode):
    sat_paths, unsat_paths = [], []
    for root, dirs, files in os.walk(path_to_dataset_root, topdown=True):
        if METADATA_PATH in dirs:
            dirs.remove(METADATA_PATH)

        for file_name in files:
            cur_path = os.path.join(root, file_name)

            if cur_path.endswith("-sat"):
                sat_paths.append(cur_path)
            elif cur_path.endswith("-unsat"):
                unsat_paths.append(cur_path)
            else:
                raise Exception(f"strange file path '{cur_path}'")

    if len(sat_paths) > SHRINK:
        sat_paths = sat_paths[:SHRINK]

    if len(unsat_paths) > SHRINK:
        unsat_paths = unsat_paths[:SHRINK]

    sat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, sat_paths)
    unsat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, unsat_paths)

    def split_data_to_train_val_test(data):
        train_ind, val_ind, test_ind = train_val_test_indices(len(data), val_qty=val_qty, test_qty=test_qty)

        return [data[i] for i in train_ind], [data[i] for i in val_ind], [data[i] for i in test_ind]

    sat_train, sat_val, sat_test = split_data_to_train_val_test(sat_paths)
    unsat_train, unsat_val, unsat_test = split_data_to_train_val_test(unsat_paths)

    sat_train, unsat_train = align_sat_unsat_sizes(sat_train, unsat_train, align_train_mode)
    sat_val, unsat_val = align_sat_unsat_sizes(sat_val, unsat_val, align_val_mode)
    sat_test, unsat_test = align_sat_unsat_sizes(sat_test, unsat_test, align_test_mode)

    train_data = sat_train + unsat_train
    val_data = sat_val + unsat_val
    test_data = sat_test + unsat_test

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data


def grouped_random_split(path_to_dataset_root, val_qty, test_qty, align_train_mode, align_val_mode, align_test_mode):
    def return_group_weight(path_to_group):
        return len(os.listdir(path_to_group))

    def calc_group_weights(path_to_dataset_root):
        groups = os.listdir(path_to_dataset_root)
        groups.remove(METADATA_PATH)

        weights = [return_group_weight(os.path.join(path_to_dataset_root, group)) for group in groups]

        return list(zip(groups, weights))

    groups = calc_group_weights(path_to_dataset_root)

    def pick_best_split(groups):
        attempts = 100_000

        groups_cnt = len(groups)
        samples_cnt = sum(g[1] for g in groups)

        need_val = int(samples_cnt * val_qty)
        need_test = int(samples_cnt * test_qty)
        need_train = samples_cnt - need_val - need_test

        best = None

        for _ in trange(attempts):
            split = np.random.randint(3, size=groups_cnt)

            train_size = sum(groups[i][1] for i in range(groups_cnt) if split[i] == 0)
            val_size = sum(groups[i][1] for i in range(groups_cnt) if split[i] == 1)
            test_size = sum(groups[i][1] for i in range(groups_cnt) if split[i] == 2)

            cur_error = (train_size - need_train) ** 2 + (val_size - need_val) ** 2 + (test_size - need_test) ** 2

            if best is None or best[0] > cur_error:
                best = (cur_error, split)

        return best[1]

    split = pick_best_split(groups)

    train_data, val_data, test_data = [], [], []

    for i in range(len(groups)):
        cur_group = os.listdir(os.path.join(path_to_dataset_root, groups[i][0]))
        cur_group = list(map(lambda sample: os.path.join(path_to_dataset_root, groups[i][0], sample), cur_group))

        if split[i] == 0:
            train_data += cur_group
        elif split[i] == 1:
            val_data += cur_group
        elif split[i] == 2:
            test_data += cur_group

    train_data = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, train_data)
    val_data = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, val_data)
    test_data = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, test_data)

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data

    """

    sat_paths, unsat_paths = [], []
    for root, dirs, files in os.walk(path_to_dataset_root, topdown=True):
        if METADATA_PATH in dirs:
            dirs.remove(METADATA_PATH)

        for file_name in files:
            cur_path = os.path.join(root, file_name)

            if cur_path.endswith("-sat"):
                sat_paths.append(cur_path)
            elif cur_path.endswith("-unsat"):
                unsat_paths.append(cur_path)
            else:
                raise Exception(f"strange file path '{cur_path}'")

    if len(sat_paths) > SHRINK:
        sat_paths = sat_paths[:SHRINK]

    if len(unsat_paths) > SHRINK:
        unsat_paths = unsat_paths[:SHRINK]

    sat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, sat_paths)
    unsat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, unsat_paths)

    def split_data_to_train_val_test(data):
        train_ind, val_ind, test_ind = train_val_test_indices(len(data), val_qty=val_qty, test_qty=test_qty)

        return [data[i] for i in train_ind], [data[i] for i in val_ind], [data[i] for i in test_ind]

    sat_train, sat_val, sat_test = split_data_to_train_val_test(sat_paths)
    unsat_train, unsat_val, unsat_test = split_data_to_train_val_test(unsat_paths)

    sat_train, unsat_train = align_sat_unsat_sizes(sat_train, unsat_train, align_train_mode)
    sat_val, unsat_val = align_sat_unsat_sizes(sat_val, unsat_val, align_val_mode)
    sat_test, unsat_test = align_sat_unsat_sizes(sat_test, unsat_test, align_test_mode)

    train_data = sat_train + unsat_train
    val_data = sat_val + unsat_val
    test_data = sat_test + unsat_test

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data
    """


def create_split(path_to_dataset_root, val_qty, test_qty, align_train_mode, align_val_mode, align_test_mode, grouped):
    if grouped:
        train_data, val_data, test_data = grouped_random_split(
            path_to_dataset_root,
            val_qty, test_qty,
            align_train_mode, align_val_mode, align_test_mode
        )

    else:
        train_data, val_data, test_data = classic_random_split(
            path_to_dataset_root,
            val_qty, test_qty,
            align_train_mode, align_val_mode, align_test_mode
        )

    print("\nstats:", flush=True)
    print(f"train: {len(train_data)}")
    print(f"val:   {len(val_data)}")
    print(f"test:  {len(test_data)}")
    print(flush=True)

    meta_path = os.path.join(path_to_dataset_root, METADATA_PATH)
    os.makedirs(meta_path, exist_ok=True)

    with open(os.path.join(meta_path, "train"), "w") as f:
        f.write("\n".join(train_data))

        if len(train_data):
            f.write("\n")

    with open(os.path.join(meta_path, "val"), "w") as f:
        f.write("\n".join(val_data))

        if len(val_data):
            f.write("\n")

    with open(os.path.join(meta_path, "test"), "w") as f:
        f.write("\n".join(test_data))

        if len(test_data):
            f.write("\n")


def get_args():
    parser = ArgumentParser(description="train/val/test splitting script")

    parser.add_argument("--ds", required=True)

    parser.add_argument("--val_qty", type=float, default=0.15)
    parser.add_argument("--test_qty", type=float, default=0.1)

    parser.add_argument("--align_train", choices=["none", "upsample", "downsample"], default="upsample")
    parser.add_argument("--align_val", choices=["none", "upsample", "downsample"], default="none")
    parser.add_argument("--align_test", choices=["none", "upsample", "downsample"], default="none")

    parser.add_argument("--grouped", action="store_true")

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    seed_everything(24)

    args = get_args()

    create_split(
        args.ds,
        args.val_qty, args.test_qty,
        args.align_train, args.align_val, args.align_test,
        args.grouped
    )
