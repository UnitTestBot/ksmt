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

from GraphReader import read_graph_by_path
from utils import train_val_test_indices, align_sat_unsat_sizes


BATCH_SIZE = 32
MAX_FORMULA_SIZE = 10000
MAX_FORMULA_DEPTH = 2500
NUM_WORKERS = 16


class GraphDataset(Dataset):
    def __init__(self, graph_data):
        """
        assert (
                len(node_sets) == len(edge_sets)
                and len(node_sets) == len(labels)
                and len(labels) == len(depths)
        )
        """

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

    SHRINK = 10 ** 10  # 1000

    if len(sat_paths) > SHRINK:
        sat_paths = sat_paths[:SHRINK]

    if len(unsat_paths) > SHRINK:
        unsat_paths = unsat_paths[:SHRINK]

    def process_paths(paths, label, data):
        for path in tqdm(paths):
            operators, edges, depth = read_graph_by_path(path, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH)

            if depth is None:
                continue

            if len(edges) == 0:
                print(f"w: formula with no edges; file '{path}'")
                continue

            data.append((operators, edges, label, depth))

    sat_data, unsat_data = [], []
    process_paths(sat_paths, 1, sat_data)
    process_paths(unsat_paths, 0, unsat_data)

    np.random.seed(24)
    sat_data, unsat_data = align_sat_unsat_sizes(sat_data, unsat_data)

    graph_data = sat_data + unsat_data
    del sat_data, unsat_data
    gc.collect()

    """
    assert (
            len(all_operators) == len(all_edges)
            and len(all_edges) == len(all_labels)
            and len(all_labels) == len(all_depths)
    )
    """

    train_ind, val_ind, test_ind = train_val_test_indices(len(graph_data))

    train_data = [graph_data[i] for i in train_ind]
    val_data = [graph_data[i] for i in val_ind]
    test_data = [graph_data[i] for i in test_ind]

    print("\nstats:")
    print(f"overall: {sum(it[2] for it in graph_data) / len(graph_data)} | {len(graph_data)}")
    print(f"train:   {sum(it[2] for it in train_data) / len(train_data)} | {len(train_data)}")
    print(f"val:     {sum(it[2] for it in val_data) / len(val_data)} | {len(val_data)}")
    print(f"test:    {sum(it[2] for it in test_data) / len(test_data)} | {len(test_data)}")
    print("\n", flush=True)

    """
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
    """

    # assert (len(train_operators) == len(train_edges) and len(train_edges) == len(train_labels))
    # assert (len(val_operators) == len(val_edges) and len(val_edges) == len(val_labels))
    # assert (len(test_operators) == len(test_edges) and len(test_edges) == len(test_labels))

    print("del start")
    del graph_data
    gc.collect()
    print("del end")

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

    """
    def transform_data(data):
        for i in range(len(data)):
            data[i][0] = encoder.transform(np.array(data[i][0]).reshape(-1, 1))
    """

    def transform(data_for_one_graph):
        nodes, edges, label, depth = data_for_one_graph
        nodes = encoder.transform(np.array(nodes).reshape(-1, 1))
        return nodes, edges, label, depth

    print("transform start")
    #transform_data(train_data)
    #transform_data(val_data)
    #transform_data(test_data)

    train_data = list(map(transform, train_data))
    val_data = list(map(transform, val_data))
    test_data = list(map(transform, test_data))

    #train_operators = list(map(transform, train_operators))
    #val_operators = list(map(transform, val_operators))
    #test_operators = list(map(transform, test_operators))
    print("transform end")

    train_ds = GraphDataset(train_data)
    val_ds = GraphDataset(val_data)
    test_ds = GraphDataset(test_data)

    #train_ds = GraphDataset(train_operators, train_edges, train_labels, train_depths)
    #val_ds = GraphDataset(val_operators, val_edges, val_labels, val_depths)
    #test_ds = GraphDataset(test_operators, test_edges, test_labels, test_depths)

    return (
        DataLoader(train_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True),
        DataLoader(val_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS),
        DataLoader(test_ds.graphs, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    )
