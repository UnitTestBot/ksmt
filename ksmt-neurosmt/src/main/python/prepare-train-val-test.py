#!/usr/bin/python3

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from pytorch_lightning import seed_everything

from GraphReader import read_graph_by_path
from GraphDataloader import MAX_FORMULA_SIZE, MAX_FORMULA_DEPTH, SHRINK, METADATA_PATH
from utils import train_val_test_indices, align_sat_unsat_sizes


def create_split(path_to_dataset):
    sat_paths, unsat_paths = [], []
    for root, dirs, files in os.walk(path_to_dataset, topdown=True):
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

    def process_paths(paths):
        correct_paths = []
        for path in tqdm(paths):
            operators, edges, depth = read_graph_by_path(path, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH)

            if depth is None:
                continue

            if len(edges) == 0:
                print(f"w: ignoring formula without edges; file '{path}'")
                continue

            correct_paths.append(os.path.relpath(path, path_to_dataset))

        return correct_paths

    sat_paths = process_paths(sat_paths)
    unsat_paths = process_paths(unsat_paths)

    sat_paths, unsat_paths = align_sat_unsat_sizes(sat_paths, unsat_paths)

    def split_data(data):
        train_ind, val_ind, test_ind = train_val_test_indices(len(data))

        return [data[i] for i in train_ind], [data[i] for i in val_ind], [data[i] for i in test_ind]

    sat_train, sat_val, sat_test = split_data(sat_paths)
    unsat_train, unsat_val, unsat_test = split_data(unsat_paths)

    train_data = sat_train + unsat_train
    val_data = sat_val + unsat_val
    test_data = sat_test + unsat_test

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    print("\nstats:", flush=True)
    print(f"train: {len(train_data)}")
    print(f"val:   {len(val_data)}")
    print(f"test:  {len(test_data)}")
    print(flush=True)

    meta_path = os.path.join(path_to_dataset, METADATA_PATH)
    os.makedirs(meta_path, exist_ok=True)

    with open(os.path.join(meta_path, "train"), "w") as f:
        f.write("\n".join(train_data) + "\n")

    with open(os.path.join(meta_path, "val"), "w") as f:
        f.write("\n".join(val_data) + "\n")

    with open(os.path.join(meta_path, "test"), "w") as f:
        f.write("\n".join(test_data) + "\n")


def get_args():
    parser = ArgumentParser(description="train/val/test splitting script")
    parser.add_argument("--ds", required=True)

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    seed_everything(24)

    args = get_args()

    create_split(args.ds)
