import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_hidden, n_timesteps, n_features):
        super().__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=n_hidden)
        self.fc_ = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.fc__ = nn.Linear(in_features=n_timesteps * n_hidden, out_features=1)

    def forward(self, x: torch.Tensor):
        x = torch.tensor(data=x, dtype=torch.float32)
        o = self.fc(x)
        o = self.fc_(o)
        o = o.view(size=(o.size(0), -1))
        o = self.fc__(o)
        o = o.squeeze()
        return o
