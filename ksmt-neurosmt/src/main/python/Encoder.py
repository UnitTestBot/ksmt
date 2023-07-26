import torch.nn as nn

from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(2000, hidden_dim)
        #self.conv = GatedGraphConv(64, num_layers=1, aggr="mean")
        self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        #self.conv = GATConv(64, 64, add_self_loops=False)

    def forward(self, data):
        x, edge_index, depth = data.x, data.edge_index, data.depth

        x = self.embedding(x.squeeze())

        depth = depth.max()
        for i in range(depth):
            x = self.conv(x, edge_index)

        return x
