import torch.nn as nn


DECODER_LAYERS = 4


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.act = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(DECODER_LAYERS - 1)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
            x = self.act(x)

        x = self.out(x)

        return x
