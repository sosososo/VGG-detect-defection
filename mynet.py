import torch.nn as nn
import math
import torch
from config import *


__all__ = [
    'net5', 'net7', 'net9', 'net11', 'net13', 
]


class mynet(nn.Module):

    def __init__(self, features, num_classes=2):
        super(mynet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        self.num_classes = num_classes
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = self.lastclassifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 'M', 256, 'M', 512, 'M'],
    'B': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 512, 'M'],

}

def net7(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "F")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mynet(make_layers(cfg['A']), **kwargs)
    # print('VGG 11 model:')
    # print(model)
    '''
    if pretrained:
        if model.num_classes!=1000:
            pretrained_dick=torch.load(model_param_location['vgg11'])
            model_dict=model.state_dict()
            pretrained_dick={k:v for k,v in pretrained_dick.items() if k in model_dict}
            model_dict.update(pretrained_dick)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(model_param_location['vgg11']))
    '''
    return model




def net9(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "F")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = mynet(make_layers(cfg['B']), **kwargs)
    # print('VGG 11 model:')
    # print(model)
    if pretrained:
        if model.num_classes!=1000:
            pretrained_dick=torch.load(model_param_location['vgg11'])
            model_dict=model.state_dict()
            pretrained_dick={k:v for k,v in pretrained_dick.items() if k in model_dict}
            model_dict.update(pretrained_dick)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(model_param_location['vgg11']))
    return model

