import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder

EMBEDDING_DIM = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(hidden_dim=EMBEDDING_DIM)
        self.decoder = Decoder(hidden_dim=EMBEDDING_DIM)

    def forward(self, data):
        data = self.encoder(data)
        data = self.decoder(data)

        return data
