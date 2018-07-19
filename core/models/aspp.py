import torch
import torch.nn as nn


class ASPP_simple(nn.Module):
    def __init__(self, in_channels, channels=512, kernel_size=3, dilation_series=[6, 12, 18, 24, 1], bn_param=dict(), **kwargs):
        super(ASPP_simple, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.feature_dim = channels
        for dilation in dilation_series:
            padding = dilation * int((kernel_size - 1) / 2)
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        self.out_conv2d = nn.Conv2d(len(dilation_series) * channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_bn2d = nn.BatchNorm2d(channels, **bn_param)

    def forward(self, x):
        outs = []
        outs.append(self.conv2d_list[0](x))
        for i in range(1, len(self.conv2d_list)):
            outs.append(self.conv2d_list[i](x))
        out = torch.cat(tuple(outs), dim=1)
        out = self.out_conv2d(out)
        out = self.out_bn2d(out)
        out = self.relu(out)
        return out
