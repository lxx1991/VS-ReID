import torch.nn as nn

__all__ = ['SenseResNet', 'sense_resnet']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_param=dict()):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes, **bn_param)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, **bn_param)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_param=dict()):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, **bn_param)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, **bn_param)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, **bn_param)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SenseResNet(nn.Module):
    def __init__(self, block, layers, config, isflow=False):
        super(SenseResNet, self).__init__()
        if isflow:
            self.input_mean = [0.0, 0.0]
            self.input_std = [1.0, 1.0]
        else:
            self.input_mean = [104, 117, 123]
            self.input_std = [1, 1, 1]
        self.inplanes = 128
        bn_param = config.bn_param
        stride = config.layers_stride
        dilation = config.layers_dilation
        self.relu = nn.ReLU(inplace=True)
        self.prob_encode = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        if isflow:
            self.conv1_1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64, **bn_param)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64, **bn_param)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(128, **bn_param)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(0, 4):
            setattr(self, 'layer{}'.format(i + 1), self._make_layer(block, 64 * 2**i, layers[i], stride=stride[i], dilation=dilation[i], bn_param=bn_param))

        self.feature_dim = self.inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.09)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_param=dict()):
        downsample = None
        if isinstance(dilation, int):
            dilation = [dilation for _ in range(blocks)]

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, **bn_param))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation[0], downsample, bn_param))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation[i], bn_param=bn_param))

        return nn.Sequential(*layers)

    def forward(self, x, p):
        x = self.conv1_1(x) + self.prob_encode(p)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def sense_resnet(config, isflow=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SenseResNet(Bottleneck, [3, 4, 23, 3], config=config, isflow=isflow)
    return model
