import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        data = self.lin1(data)
        data = self.act(data)
        data = self.lin2(data)

        return data
