import os
import gc
import itertools

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from GraphReader import read_graph_from_file
from utils import train_val_test_indices, align_sat_unsat_sizes


# torch.manual_seed(0)

BATCH_SIZE = 1  # 32


class GraphDataset(Dataset):
    def __init__(self, node_sets, edge_sets, labels, depths):
        assert (
                len(node_sets) == len(edge_sets)
                and len(node_sets) == len(labels)
                and len(labels) == len(depths)
        )

        self.graphs = [Graph(
            x=torch.tensor(nodes),
            edge_index=torch.tensor(edges).t(),
            y=torch.tensor([[label]], dtype=torch.float),
            depth=depth
        ) for nodes, edges, label, depth in zip(node_sets, edge_sets, labels, depths)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def load_data(path_to_data):
    sat_paths, unsat_paths = [], []
    for it in tqdm(os.walk(path_to_data)):
        for file_name in tqdm(it[2]):
            cur_path = os.path.join(it[0], file_name)
            if cur_path.endswith("-sat"):
                sat_paths.append(cur_path)
            elif cur_path.endswith("-unsat"):
                unsat_paths.append(cur_path)
            else:
                raise Exception(f"strange file path '{cur_path}'")

    np.random.seed(24)
    sat_paths, unsat_paths = align_sat_unsat_sizes(sat_paths, unsat_paths)

    all_operators, all_edges, all_labels, all_depths = [], [], [], []

    for cur_path in tqdm(sat_paths):
        operators, edges, depth = read_graph_from_file(cur_path)

        if len(edges) == 0:
            print(f"w: formula with no edges; file '{cur_path}'")
            continue

        all_operators.append(operators)
        all_edges.append(edges)
        all_labels.append(1)
        all_depths.append(depth)

    for cur_path in tqdm(unsat_paths):
        operators, edges, depth = read_graph_from_file(cur_path)

        if len(edges) == 0:
            print(f"w: formula with no edges; file '{cur_path}'")
            continue

        all_operators.append(operators)
        all_edges.append(edges)
        all_labels.append(0)
        all_depths.append(depth)

    """
    for it in tqdm(os.walk(path_to_data)):
        for file_name in tqdm(it[2]):
            cur_path = os.path.join(it[0], file_name)

            operators, edges = read_graph_from_file(cur_path)

            if len(edges) == 0:
                print(f"w: formula with no edges; file '{cur_path}'")
                continue

            all_operators.append(operators)
            all_edges.append(edges)
            if cur_path.endswith("-sat"):
                all_labels.append(1)
            elif cur_path.endswith("-unsat"):
                all_labels.append(0)
            else:
                raise Exception(f"strange file path '{cur_path}'")
    """

    assert (
            len(all_operators) == len(all_edges)
            and len(all_edges) == len(all_labels)
            and len(all_labels) == len(all_depths)
    )

    train_ind, val_ind, test_ind = train_val_test_indices(len(all_operators))

    train_operators = [all_operators[i] for i in train_ind]
    train_edges = [all_edges[i] for i in train_ind]
    train_labels = [all_labels[i] for i in train_ind]
    train_depths = [all_depths[i] for i in train_ind]

    val_operators = [all_operators[i] for i in val_ind]
    val_edges = [all_edges[i] for i in val_ind]
    val_labels = [all_labels[i] for i in val_ind]
    val_depths = [all_depths[i] for i in val_ind]

    test_operators = [all_operators[i] for i in test_ind]
    test_edges = [all_edges[i] for i in test_ind]
    test_labels = [all_labels[i] for i in test_ind]
    test_depths = [all_depths[i] for i in test_ind]

    # assert (len(train_operators) == len(train_edges) and len(train_edges) == len(train_labels))
    # assert (len(val_operators) == len(val_edges) and len(val_edges) == len(val_labels))
    # assert (len(test_operators) == len(test_edges) and len(test_edges) == len(test_labels))

    del all_operators, all_edges, all_labels, all_depths
    gc.collect()

    encoder = OrdinalEncoder(
        dtype=np.int,
        handle_unknown="use_encoded_value", unknown_value=-1,
        encoded_missing_value=-2
    )

    encoder.fit(np.array(list(itertools.chain(*train_operators))).reshape(-1, 1))

    def transform(op_list):
        return encoder.transform(np.array(op_list).reshape(-1, 1))

    train_operators = list(map(transform, train_operators))
    val_operators = list(map(transform, val_operators))
    test_operators = list(map(transform, test_operators))

    train_ds = GraphDataset(train_operators, train_edges, train_labels, train_depths)
    val_ds = GraphDataset(val_operators, val_edges, val_labels, val_depths)
    test_ds = GraphDataset(test_operators, test_edges, test_labels, test_depths)

    return (
        DataLoader(train_ds.graphs, batch_size=BATCH_SIZE),
        DataLoader(val_ds.graphs, batch_size=BATCH_SIZE),
        DataLoader(test_ds.graphs, batch_size=BATCH_SIZE)
    )
