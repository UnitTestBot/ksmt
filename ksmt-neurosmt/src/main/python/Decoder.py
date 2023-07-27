import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.act = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        #print("=" * 12)
        #print(data.shape)

        for layer in self.linears:
            x = layer(x)
            x = self.act(x)

        x = self.out(x)

        #print(data.shape)

        return x
