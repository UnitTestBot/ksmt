import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv


EMBEDDING_DIM = 2000


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(EMBEDDING_DIM, hidden_dim)
        self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)

        #self.conv = GATConv(64, 64, add_self_loops=False)

    def forward(self, node_labels, edges, depths, root_ptrs):
        # x, edge_index, depth = data.x, data.edge_index, data.depth
        # edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long).to(edge_index.get_device())

        node_features = self.embedding(node_labels.squeeze())

        depth = depths.max()
        node_features += depth * 0.0
        for i in range(depth):
            node_features = self.conv(node_features, edges)

        node_features = node_features[root_ptrs[1:] - 1]

        return node_features
