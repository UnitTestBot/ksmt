#!/usr/bin/python3

import sys
import os; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import time

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter


from GraphReader import read_graph_from_file
from GraphDataloader import load_data


import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from Encoder import Encoder
from Decoder import Decoder
from Model import Model

from sklearn.metrics import accuracy_score, classification_report


"""
class GCNConv1(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow="source_to_target")  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(1000, 16)
        self.conv1 = GCNConv(16, 16, add_self_loops=False)
        self.conv2 = GCNConv(16, 1, add_self_loops=False)

        #self.conv1 = GCNConv(16, 1)
        #self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x.squeeze())
        #x[-2] = 1.23
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
"""


if __name__ == "__main__":
    tr, va, te = load_data(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p is not None and p.requires_grad], lr=1e-4)

    def calc_grad_norm():
        grads = [
            p.grad.detach().flatten() for p in model.parameters() if p.grad is not None and p.requires_grad
        ]
        return torch.cat(grads).norm().item()

    for p in model.parameters():
        assert (p.requires_grad)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in trange(100):
        model.train()
        for batch in tqdm(tr):
            optimizer.zero_grad()
            batch = batch.to(device)

            out = model(batch)
            out = out[batch.ptr[:-1]]

            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            #loss = criterion(out, batch.y)
            loss.backward()

            optimizer.step()

        print("\n", flush=True)
        print(f"grad norm: {calc_grad_norm()}")

        def validate(dl):
            model.eval()

            #all_ans, correct_ans = 0, 0
            answers, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)
            losses = []
            with torch.no_grad():
                for batch in tqdm(dl):
                    batch = batch.to(device)

                    out = model(batch)
                    out = out[batch.ptr[:-1]]
                    loss = F.binary_cross_entropy_with_logits(out, batch.y)

                    out = F.sigmoid(out)
                    out = (out > 0.5)

                    answers = torch.cat((answers, out))
                    targets = torch.cat((targets, batch.y.to(torch.int).to(torch.bool)))
                    losses.append(loss.item())

                    #all_ans += len(batch.y)
                    #out: torch.Tensor = (batch.y.to(torch.int) == out.to(torch.bool))
                    #correct_ans += out.int().sum().item()

            answers = torch.flatten(answers).detach().cpu().numpy()
            targets = torch.flatten(targets).detach().cpu().numpy()

            #print(f"\n{correct_ans / all_ans}")
            print(flush=True)
            print(f"mean loss: {np.mean(losses)}")
            print(f"acc: {accuracy_score(targets, answers)}", flush=True)
            print(classification_report(targets, answers, digits=3, zero_division=0.0), flush=True)

        print()
        print("train:")
        validate(tr)
        print("val:")
        validate(va)
        print()

















    #for batch in tr:
    #    print(batch.num_graphs, batch)
        #print(batch.ptr)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1):
        for batch in tr:
            #optimizer.zero_grad()
            out = model(batch)
            #print(out[0].item(), out[-1].item(), batch[0].num_nodes, batch[0].num_edges, batch[0].depth.item())
            print(batch[0].depth.item(), batch[0].num_nodes, batch[0].num_edges)

            out = scatter(out, batch.batch, dim=0, reduce='mean')
            #loss = F.mse_loss(out, batch.y)
            #loss.backward()
            #optimizer.step()

    """

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
