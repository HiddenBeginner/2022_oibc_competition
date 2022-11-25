import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(WaveNet, self).__init__()
        self.conv_block1 = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding='same', dilation=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=8)
        ])

        self.conv_block2 = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding='same', dilation=8)
        ])

        self.conv_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

        orthogonal_init(self.conv_block1)
        orthogonal_init(self.conv_block2)
        orthogonal_init(self.conv_out)

    def forward(self, x):
        for layer in self.conv_block1:
            x = F.relu(layer(x))

        for layer in self.conv_block2:
            x = F.relu(layer(x))

        x = self.conv_out(x)
        x = torch.transpose(x, 1, 2)

        return F.relu(x)


def orthogonal_init(layer, nonlinearity="relu"):
    """
    https://github.com/kakaoenterprise/JORLDY/blob/master/jorldy/core/network/utils.py#L109
    """
    gain = torch.nn.init.calculate_gain(nonlinearity)
    if isinstance(layer, nn.ModuleList):
        for l in layer:
            torch.nn.init.orthogonal_(l.weight.data, gain)
            torch.nn.init.zeros_(l.bias.data)
    else:
        torch.nn.init.orthogonal_(layer.weight.data, gain)
        torch.nn.init.zeros_(layer.bias.data)


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class CausalWaveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(CausalWaveNet, self).__init__()
        self.conv_block1 = nn.ModuleList([
            CausalConv1d(input_dim, hidden_dim, kernel_size=2, dilation=1),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=4),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=8)
        ])

        self.conv_block2 = nn.ModuleList([
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=1),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=4),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=8)
        ])

        self.conv_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

        orthogonal_init(self.conv_block1)
        orthogonal_init(self.conv_block2)
        orthogonal_init(self.conv_out)

    def forward(self, x):
        for layer in self.conv_block1:
            x = F.relu(layer(x))
            x = x[:, :, :-layer.padding[0]]

        for layer in self.conv_block2:
            x = F.relu(layer(x))
            x = x[:, :, :-layer.padding[0]]

        x = self.conv_out(x)
        x = torch.transpose(x, 1, 2)

        return F.relu(x)
