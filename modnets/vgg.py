import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import sys
sys.path.append('../')
import modnets.layers as nl

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]



class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class VGG(nn.Module):

    def __init__(self, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((7, 7)),
            nl.ElementWiseLinear(
                512 * 7 * 7, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),###mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', make_model=True, original=False, num_class=1000
            nn.ReLU(True),
            nn.Dropout(),
            nl.ElementWiseLinear(
                4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            # nl.ElementWiseLinear(4096, num_classes),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, mask_init, mask_scale, threshold_fn, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nl.ElementWiseConv2d(
                in_channels, v, kernel_size=3, padding=1,
                mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.AdaptiveAvgPool2d((7, 7))]
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D")."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn, **kwargs)
    return model


def vgg16_bn(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn, batch_norm=True),
                mask_init, mask_scale, threshold_fn, **kwargs)  # , **kwargs
    return model
