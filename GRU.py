import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, n_hidden, n_features):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=n_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, x):
        x = torch.tensor(data=x, dtype=torch.float32)
        o = self.gru(x)[0][:, -1, :]
        o = self.fc(o)
        o = o.squeeze()
        return o
