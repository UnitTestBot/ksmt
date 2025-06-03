import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv

from GlobalConstants import EMBEDDINGS_CNT


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(EMBEDDINGS_CNT, hidden_dim)
        self.conv = SAGEConv(hidden_dim, hidden_dim, "mean", root_weight=True, project=True)

        # other options (not supported yet)
        # self.conv = GATConv(hidden_dim, hidden_dim, add_self_loops=True)       # this can't be exported to ONNX
        # self.conv = TransformerConv(hidden_dim, hidden_dim, root_weight=True)  # this can't be exported to ONNX

    def forward(
            self, node_labels: torch.Tensor, edges: torch.Tensor, depths: torch.Tensor, root_ptrs: torch.Tensor
    ) -> torch.Tensor:
        """
        encoder forward pass

        :param node_labels: torch.Tensor of shape [number of nodes in batch, 1] (dtype=int32)
        :param edges: torch.Tensor of shape [2, number of edges in batch] (dtype=int64)
        :param depths: torch.Tensor of shape [number of nodes in batch] (dtype=int32)
        :param root_ptrs: torch.Tensor of shape [batch size + 1] (dtype=int32) --
                pointers to root of graph for each expression
        :return: torch.Tensor of shape [batch size, hidden dimension size] (dtype=float) --
                embeddings for each expression
        """

        node_features = self.embedding(node_labels.squeeze())

        depth = depths.max()
        for i in range(1, depth + 1):
            mask = (depths == i)
            new_features = self.conv(node_features, edges)

            node_features = torch.clone(node_features)
            node_features[mask] = new_features[mask]

        node_features = node_features[root_ptrs[1:] - 1]

        return node_features
