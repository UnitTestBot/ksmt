import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder


class Model(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim)

    def forward(
            self, node_labels: torch.Tensor, edges: torch.Tensor, depths: torch.Tensor, root_ptrs: torch.Tensor
    ) -> torch.Tensor:
        """
        model full forward pass (encoder + decoder)

        :param node_labels: torch.Tensor of shape [number of nodes in batch, 1] (dtype=int32)
        :param edges: torch.Tensor of shape [2, number of edges in batch] (dtype=int64)
        :param depths: torch.Tensor of shape [number of nodes in batch] (dtype=int32)
        :param root_ptrs: torch.Tensor of shape [batch size + 1] (dtype=int32) --
                pointers to root of graph for each expression
        :return: torch.Tensor of shape [batch size, 1] (dtype=float) --
                each element is a logit for probability of formula to be SAT
        """

        x = self.encoder(node_labels, edges, depths, root_ptrs)
        x = self.decoder(x)

        return x
