import torch
import torch.nn as nn


class CausalResize(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        return x[..., : - self.padding_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, n_features, n_filters, filter_size, dilation=1, dropout_rate=0.1):
        super().__init__()
        self.padding_size = filter_size - 1
        # in_channels: 输入特征维度, out_channels: 输出通道数，卷积核数量
        #self.conv = weight_norm(nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=filter_size,
        #                                  stride=1, padding=self.padding_size, dilation=dilation))
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=filter_size,
                              stride=1, padding=self.padding_size, dilation=dilation)
        self.resize = CausalResize(padding_size=self.padding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        #self.conv_ = weight_norm(nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size,
        #                                   stride=1, padding=self.padding_size, dilation=dilation))
        self.conv_ = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size,
                               stride=1, padding=self.padding_size, dilation=dilation)
        self.resize_ = CausalResize(padding_size=self.padding_size)
        self.relu_ = nn.ReLU()
        self.dropout_ = nn.Dropout(p=dropout_rate)

        self.net = nn.Sequential(self.conv, self.resize, self.relu, self.dropout)
        #                         self.conv_, self.resize_, self.relu_, self.dropout_)

        self.conv_residual = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=1) \
            if n_features != n_filters else None
        self.relu__ = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x_ = self.net(x)
        residual = x if self.conv_residual is None else self.conv_residual(x)
        return self.relu__(x_ + residual).permute(0, 2, 1)


class TCN(nn.Module):
    def __init__(self, n_features, n_filters, n_timesteps, filter_size):
        super().__init__()
        self.tcn = TCNBlock(n_features=n_features, n_filters=n_filters, filter_size=filter_size)
        self.fc = nn.Linear(in_features=n_timesteps * n_filters, out_features=1)

    def forward(self, x):
        x = torch.tensor(data=x, dtype=torch.float32)
        o = self.tcn(x)
        o = torch.reshape(o, shape=(o.size(0), -1))  # Flatten
        o = self.fc(o)
        o = o.squeeze()
        return o


