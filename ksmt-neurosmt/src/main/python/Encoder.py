import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv


EMBEDDINGS_CNT = 2000


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(EMBEDDINGS_CNT, hidden_dim)
        self.conv = SAGEConv(hidden_dim, hidden_dim, "mean", root_weight=True, project=True)

        # self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        # self.conv = GATConv(hidden_dim, hidden_dim, add_self_loops=False)       # this can't be exported to ONNX
        # self.conv = TransformerConv(hidden_dim, hidden_dim, root_weight=False)  # this can't be exported to ONNX

    def forward(self, node_labels, edges, depths, root_ptrs):
        node_features = self.embedding(node_labels.squeeze())

        depth = depths.max()
        for i in range(1, depth + 1):
            mask = (depths == i)
            new_features = self.conv(node_features, edges)

            node_features = torch.clone(node_features)
            node_features[mask] = new_features[mask]

        node_features = node_features[root_ptrs[1:] - 1]

        return node_features
