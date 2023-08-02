import os
import gc
import itertools

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from GraphReader import read_graph_by_path
from utils import train_val_test_indices, align_sat_unsat_sizes


BATCH_SIZE = 32
MAX_FORMULA_SIZE = 10000
MAX_FORMULA_DEPTH = 2500
NUM_WORKERS = 16
SHRINK = 10 ** (int(os.environ["SHRINK"]) if "SHRINK" in os.environ else 10)
METADATA_PATH = "__meta"


class GraphDataset(Dataset):
    def __init__(self, graph_data):

        self.graphs = [Graph(
            x=torch.tensor(nodes),
            edge_index=torch.tensor(edges).t(),
            y=torch.tensor([[label]], dtype=torch.float),
            depth=depth
        ) for nodes, edges, label, depth in graph_data]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def load_data(paths_to_datasets, target):
    data = []

    print(f"loading {target}")
    for path_to_dataset in paths_to_datasets:
        print(f"loading data from '{path_to_dataset}'")

        with open(os.path.join(path_to_dataset, METADATA_PATH, target), "r") as f:
            for path_to_sample in tqdm(list(f.readlines())):
                path_to_sample = os.path.join(path_to_dataset, path_to_sample.strip())

                operators, edges, depth = read_graph_by_path(
                    path_to_sample, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH
                )

                if depth is None:
                    continue

                if len(edges) == 0:
                    print(f"w: ignoring formula without edges; file '{path_to_sample}'")
                    continue

                label = None
                if path_to_sample.endswith("-sat"):
                    label = 1
                elif path_to_sample.endswith("-unsat"):
                    label = 0
                else:
                    raise Exception(f"strange file path '{path_to_sample}'")

                data.append((operators, edges, label, depth))

    return data


def get_dataloader(paths_to_datasets, target, path_to_ordinal_encoder):
    print(f"creating dataloader for {target}")

    data = load_data(paths_to_datasets, target)

    print("loading encoder")
    encoder = joblib.load(path_to_ordinal_encoder)

    def transform(data_for_one_sample):
        nodes, edges, label, depth = data_for_one_sample
        nodes = encoder.transform(np.array(nodes).reshape(-1, 1))
        return nodes, edges, label, depth

    print("transforming")
    data = list(map(transform, data))

    print("creating dataset")
    ds = GraphDataset(data)

    print("constructing dataloader\n", flush=True)
    return DataLoader(
        ds.graphs,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=(target == "train"), drop_last=(target == "train")
    )


"""
if __name__ == "__main__":
    print(get_dataloader(["../BV"], "train", "../enc.bin"))
    print(get_dataloader(["../BV"], "val", "../enc.bin"))
    print(get_dataloader(["../BV"], "test", "../enc.bin"))
"""


# deprecated?
def load_data_from_scratch(path_to_full_dataset):
    return (
        get_dataloader(["../BV"], "train", "../enc.bin"),
        get_dataloader(["../BV"], "val", "../enc.bin"),
        get_dataloader(["../BV"], "test", "../enc.bin")
    )

    sat_paths, unsat_paths = [], []
    for it in tqdm(os.walk(path_to_full_dataset)):
        for file_name in tqdm(it[2]):
            cur_path = os.path.join(it[0], file_name)

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

    def process_paths(paths, data, label):
        for path in tqdm(paths):
            operators, edges, depth = read_graph_by_path(path, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH)

            if depth is None:
                continue

            if len(edges) == 0:
                print(f"w: ignoring formula without edges; file '{path}'")
                continue

            data.append((operators, edges, label, depth))

    sat_data, unsat_data = [], []
    process_paths(sat_paths, sat_data, label=1)
    process_paths(unsat_paths, unsat_data, label=0)

    sat_data, unsat_data = align_sat_unsat_sizes(sat_data, unsat_data)

    def split_data(data):
        train_ind, val_ind, test_ind = train_val_test_indices(len(data))

        return [data[i] for i in train_ind], [data[i] for i in val_ind], [data[i] for i in test_ind]

    print("train/val/test split start")

    sat_train, sat_val, sat_test = split_data(sat_data)
    unsat_train, unsat_val, unsat_test = split_data(unsat_data)
    del sat_data, unsat_data
    gc.collect()

    train_data = sat_train + unsat_train
    val_data = sat_val + unsat_val
    test_data = sat_test + unsat_test
    del sat_train, sat_val, sat_test, unsat_train, unsat_val, unsat_test
    gc.collect()

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    print("train/val/test split end")

    print("\nstats:")
    print(f"train:   {sum(it[2] for it in train_data) / len(train_data)} | {len(train_data)}")
    print(f"val:     {sum(it[2] for it in val_data) / len(val_data)} | {len(val_data)}")
    print(f"test:    {sum(it[2] for it in test_data) / len(test_data)} | {len(test_data)}")
    print("\n", flush=True)

    encoder = OrdinalEncoder(
        dtype=int,
        handle_unknown="use_encoded_value", unknown_value=1999,
        encoded_missing_value=1998
    )

    print("enc fit start")
    encoder.fit(np.array(list(itertools.chain(
        *(list(zip(*train_data))[0])
    ))).reshape(-1, 1))
    print("enc fit end")

    def transform(data_for_one_graph):
        nodes, edges, label, depth = data_for_one_graph
        nodes = encoder.transform(np.array(nodes).reshape(-1, 1))
        return nodes, edges, label, depth

    print("transform start")
    train_data = list(map(transform, train_data))
    val_data = list(map(transform, val_data))
    test_data = list(map(transform, test_data))
    print("transform end")

    print("create dataset start")
    train_ds = GraphDataset(train_data)
    val_ds = GraphDataset(val_data)
    test_ds = GraphDataset(test_data)
    print("create dataset end")

    try:
        print("create dataloader start")
        return (
            DataLoader(train_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True),
            DataLoader(val_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False),
            DataLoader(test_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
        )

    finally:
        print("create dataloader end\n", flush=True)
