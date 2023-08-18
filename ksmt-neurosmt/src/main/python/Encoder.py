import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv


EMBEDDINGS_CNT = 2000


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(EMBEDDINGS_CNT, hidden_dim)
        self.conv = SAGEConv(hidden_dim, hidden_dim, "mean", root_weight=False, project=True)

        # self.conv = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
        # self.conv = TransformerConv(hidden_dim, hidden_dim, root_weight=False)
        # self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)

    def forward(self, node_labels, edges, depths, root_ptrs):
        # x, edge_index, depth = data.x, data.edge_index, data.depth
        # edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long).to(edge_index.get_device())

        node_features = self.embedding(node_labels.squeeze())

        depth = depths.max()
        for i in range(depth):
            node_features = self.conv(node_features, edges)

        node_features = node_features[root_ptrs[1:] - 1]

        return node_features
