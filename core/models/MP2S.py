import torch
from torch import nn

from core.models import backbones
from core.models.aspp import ASPP_simple


def convert_bn(model, training):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(training)


class MP2S(nn.Module):

    def __init__(self, config):
        super(MP2S, self).__init__()
        self.config = config
        self._prepare_backbone(config.backbone)
        self._prepare_aspp(config.aspp)
        self._prepare_classifier(config.classifier)
        self.dropout = nn.Dropout2d(0.1)
        self.relu = nn.ReLU(inplace=True)

    def _prepare_backbone(self, config):
        self.backbone = getattr(backbones, config.type)(pretrained=False, config=config)
        self.flow_backbone = getattr(backbones, config.type)(pretrained=False, config=config, isflow=True)
        self.input_mean = self.backbone.input_mean
        self.input_std = self.backbone.input_std

    def _prepare_aspp(self, config):
        self.aspp = ASPP_simple(self.backbone.feature_dim, **config)
        self.flow_aspp = ASPP_simple(self.flow_backbone.feature_dim, **config)

    def _prepare_classifier(self, config):
        self.classifier = nn.Conv2d(self.aspp.feature_dim, config.num_class, kernel_size=1, stride=1, bias=True)
        self.flow_classifier = nn.Conv2d(self.flow_aspp.feature_dim, config.num_class, kernel_size=1, stride=1, bias=True)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(MP2S, self).train(mode)

        if mode:
            convert_bn(self.backbone, self.config.backbone.bn_training)
            convert_bn(self.aspp, self.config.aspp.bn_training)
            convert_bn(self.classifier, self.config.classifier.bn_training)

            convert_bn(self.flow_backbone, self.config.backbone.bn_training)
            convert_bn(self.flow_aspp, self.config.aspp.bn_training)
            convert_bn(self.flow_classifier, self.config.classifier.bn_training)

    def get_optim_policies(self):
        outs = []
        outs.extend(self.get_module_optim_policies(
            self.backbone,
            self.config.backbone,
            'backbone',
        ))
        outs.extend(self.get_module_optim_policies(
            self.aspp,
            self.config.aspp,
            'aspp',
        ))
        outs.extend(self.get_module_optim_policies(
            self.classifier,
            self.config.classifier,
            'classifier',
        ))

        outs.extend(self.get_module_optim_policies(
            self.flow_backbone,
            self.config.backbone,
            'flow_backbone',
        ))
        outs.extend(self.get_module_optim_policies(
            self.flow_aspp,
            self.config.aspp,
            'flow_aspp',
        ))
        outs.extend(self.get_module_optim_policies(
            self.flow_classifier,
            self.config.classifier,
            'flow_classifier',
        ))
        return outs

    def get_module_optim_policies(self, module, config, prefix):
        weight = []
        bias = []
        bn = []

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                bn.extend(list(m.parameters()))

        return [
            {
                'params': weight,
                'lr_mult': config.mult_conv_w[0],
                'decay_mult': config.mult_conv_w[1],
                'name': prefix + " weight"
            },
            {
                'params': bias,
                'lr_mult': config.mult_conv_b[0],
                'decay_mult': config.mult_conv_b[1],
                'name': prefix + " bias"
            },
            {
                'params': bn,
                'lr_mult': config.mult_bn[0],
                'decay_mult': config.mult_bn[1],
                'name': prefix + " bn scale/shift"
            },
        ]

    def forward(self, x, f, p):
        input_size = x.size()[2:4]
        x = x.clone()
        for i in range(3):
            x[:, i, :, :] = (x[:, i, :, :] - self.input_mean[i]) / self.input_std[i]

        x = self.backbone(x, p)
        x = self.aspp(x)
        if self.dropout is not None:
            x = self.dropout(x)
        ic = self.classifier(x)

        f = self.flow_backbone(f, p)
        f = self.flow_aspp(f)
        if self.dropout is not None:
            f = self.dropout(f)
        ic = ic + self.flow_classifier(f)

        ic = nn.functional.upsample(ic, size=input_size, mode='bilinear', align_corners=True)
        return ic
