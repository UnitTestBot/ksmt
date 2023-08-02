import os

import numpy as np
import joblib
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from GraphReader import read_graph_by_path


BATCH_SIZE = 32
MAX_FORMULA_SIZE = 10000
MAX_FORMULA_DEPTH = 2500
NUM_WORKERS = 16
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
