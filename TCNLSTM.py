import torch
import torch.nn as nn
from TCN import TCNBlock


class TCNLSTM(nn.Module):
    def __init__(self, n_features, n_filters, filter_size, dropout_rate=0.1):
        super().__init__()
        self.tcn = TCNBlock(n_features=n_features, n_filters=n_filters, filter_size=filter_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=n_filters, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=n_filters, out_features=1, bias=True)

    def forward(self, x):
        x = torch.tensor(data=x, dtype=torch.float32, requires_grad=True)
        o = self.tcn(x)
        o = self.dropout(o)
        o = self.lstm(o)[0][:, -1, :]
        o = self.fc(o)
        o = o.squeeze()
        return o
