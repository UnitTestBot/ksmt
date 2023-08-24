import os
from typing import Literal

import numpy as np
import joblib
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from GlobalConstants import BATCH_SIZE, MAX_FORMULA_SIZE, MAX_FORMULA_DEPTH, NUM_WORKERS, METADATA_PATH
from GraphReader import read_graph_by_path


# torch Dataset
class GraphDataset(Dataset):
    def __init__(self, graph_data):
        self.graphs = [Graph(
            x=torch.tensor(nodes, dtype=torch.int32),
            edge_index=torch.tensor(edges, dtype=torch.int64).t(),
            y=torch.tensor([[label]], dtype=torch.int32),
            depth=torch.tensor(depths, dtype=torch.int32)
        ) for nodes, edges, label, depths in graph_data]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


# load all samples from all datasets and return them as a list of tuples
def load_data(paths_to_datasets: list[str], target: Literal["train", "val", "test"])\
        -> list[tuple[list[str], list[tuple[int, int]], int, list[int]]]:

    data = []

    print(f"loading {target}")
    for path_to_dataset in paths_to_datasets:
        print(f"loading data from '{path_to_dataset}'")

        with open(os.path.join(path_to_dataset, METADATA_PATH, target), "r") as f:
            for path_to_sample in tqdm(list(f.readlines())):
                path_to_sample = os.path.join(path_to_dataset, path_to_sample.strip())

                operators, edges, depths = read_graph_by_path(
                    path_to_sample, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH
                )

                if operators is None or edges is None or depths is None:
                    continue

                if len(edges) == 0:
                    print(f"w: ignoring formula without edges; file '{path_to_sample}'")
                    continue

                if path_to_sample.endswith("-sat"):
                    label = 1
                elif path_to_sample.endswith("-unsat"):
                    label = 0
                else:
                    raise Exception(f"strange file path '{path_to_sample}'")

                data.append((operators, edges, label, depths))

    return data


# load samples from all datasets, transform them and return them in a Dataloader object
def get_dataloader(paths_to_datasets: list[str], target: Literal["train", "val", "test"], path_to_ordinal_encoder: str)\
        -> DataLoader:

    print(f"creating dataloader for {target}")

    print("loading data")
    data = load_data(paths_to_datasets, target)

    print(f"stats: {len(data)} overall; sat fraction is {sum(it[2] for it in data) / len(data)}")

    print("loading encoder")
    encoder = joblib.load(path_to_ordinal_encoder)

    def transform(data_for_one_sample: tuple[list[str], list[tuple[int, int]], int, list[int]])\
            -> tuple[list[str], list[tuple[int, int]], int, list[int]]:

        nodes, edges, label, depths = data_for_one_sample
        nodes = encoder.transform(np.array(nodes).reshape(-1, 1))

        return nodes, edges, label, depths

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
