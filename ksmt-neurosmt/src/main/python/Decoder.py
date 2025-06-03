import torch
import torch.nn as nn

from GlobalConstants import DECODER_LAYERS


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.act = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(DECODER_LAYERS - 1)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        decoder forward pass

        :param x: torch.Tensor of shape [batch size, hidden dimension size] (dtype=float)
        :return: torch.Tensor of shape [batch size, 1] (dtype=float) --
                each element is a logit for probability of formula to be SAT
        """

        for layer in self.linears:
            x = layer(x)
            x = self.act(x)

        x = self.out(x)

        return x
