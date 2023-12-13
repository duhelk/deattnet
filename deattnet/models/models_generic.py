'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


from packaging.version import parse as parse_version
import torchvision
import torchvision.models as models
import torch.nn as nn

from deattnet.models.netvlad import NetVLAD
from deattnet.models.layers.pooling import GeM
from deattnet.models.deattnet import DAttNet
from deattnet.models.layers.functional import L2Norm,Flatten


def get_DAttNet():
    enc_dim = 512
    model = DAttNet(dim_size=enc_dim) 
    return enc_dim, model


def get_backend(in_channels=3):
    enc_dim = 512
    if parse_version(torchvision.__version__) >= parse_version('0.13'):
        enc = models.vgg16(weights='IMAGENET1K_V1')
    else:
        enc = models.vgg16(pretrained=True)

    # replace input layer to support in channels
    if in_channels != 3:
        enc.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.kaiming_normal_(enc.features[0].weight, mode='fan_out', nonlinearity='relu')

    layers = list(enc.features.children())[:-2]

    # can use weights with no change
    if in_channels == 3:
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
    
    enc = nn.Sequential(*layers)
    return enc_dim, enc


def get_model(encoder, encoder_dim, config, append_pca_layer=False):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    if config['pooling'].lower() == 'netvlad':
        net_vlad = NetVLAD(num_clusters=int(config['num_clusters']), dim=encoder_dim,
                           vladv2=config.getboolean('vladv2'))
        nn_model.add_module('pool', net_vlad)
    elif config['pooling'].lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1, 1))
        nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif config['pooling'].lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1, 1))
        nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif config['pooling'].lower() == 'gem':
        global_pool = GeM()
        nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    else:
        raise ValueError('Unknown pooling type: ' + config['pooling'].lower())

    if append_pca_layer:
        num_pcs = int(config['num_pcs'])
        netvlad_output_dim = encoder_dim
        if config['pooling'].lower() == 'netvlad' or config['pooling'].lower() == 'patchnetvlad':
            netvlad_output_dim *= int(config['num_clusters'])

        pca_conv = nn.Conv2d(netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
        nn_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

    return nn_model
