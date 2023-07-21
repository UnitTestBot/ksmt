#!/usr/bin/python3

import sys
import os
import time

from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
#from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter


from GraphReader import read_graph_from_file
from GraphDataloader import load_data


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


if __name__ == "__main__":
    tr, va, te = load_data(sys.argv[1])
    for batch in tr:
        print(batch.num_graphs, batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in trange(50):
        for batch in tqdm(tr):
            optimizer.zero_grad()
            out = model(batch)
            out = scatter(out, batch.batch, dim=0, reduce='mean')
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()


    #print(all_edges, all_expressions)
    #print("\n\n\nFINISHED\n\n\n")
    #time.sleep(5)

    """
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    print(torch.tensor(expressions).T)
    data = Data(
        x=torch.tensor(expressions).T,
        edge_index=torch.tensor(edges)
    )
    print(data, data.is_directed())
    """
