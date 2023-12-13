from packaging.version import parse as parse_version
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch


def get_resnet18(in_channels=3):
    enc = models.resnet18(weights= models.resnet.ResNet18_Weights)
    if in_channels != 3:
        enc.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(enc.conv1.weight, mode='fan_out', nonlinearity='relu')

    layers = list(enc.children())[:-2]

    if in_channels == 3:
        for layer in layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    enc = nn.Sequential(*layers)
    return enc


def get_resnet34(in_channels=3):
    enc = models.resnet34(weights= models.resnet.ResNet34_Weights)

    if in_channels != 3:
        enc.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(enc.conv1.weight, mode='fan_out', nonlinearity='relu')

    layers = list(enc.children())[:-2]

    if in_channels == 3:
        for layer in layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    enc = nn.Sequential(*layers)
    return enc

def get_resnet50(in_channels=3):
    enc = models.resnet50(weights= models.resnet.ResNet50_Weights.DEFAULT)

    if in_channels != 3:
        enc.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(enc.conv1.weight, mode='fan_out', nonlinearity='relu')

    layers = list(enc.children())[:-2]

    if in_channels == 3:
        for layer in layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    enc = nn.Sequential(*layers)
    return enc

def get_vgg16(in_channels=3):
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
    return enc

class DepthAttention(nn.Module):
    def __init__(self, in_ch=512, out_ch=64, num_heads=4, dropout_rate=0.5):
        super(DepthAttention, self).__init__()

        self.rgb_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.rgb_bn = nn.BatchNorm2d(out_ch)
        self.rgb_act = nn.ReLU(inplace=True)

        self.dep_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.dep_bn = nn.BatchNorm2d(out_ch)
        self.dep_act = nn.ReLU(inplace=True)

        self.att_conv = nn.Conv2d(in_channels=out_ch, out_channels=num_heads, kernel_size=1)
        self.att_bn = nn.BatchNorm2d(num_heads)
        self.att_act = nn.Softmax(dim=1)

        self.out_conv = nn.Conv2d(in_channels=num_heads, out_channels=1, kernel_size=1)
        self.out_bn = nn.BatchNorm2d(1)
        self.out_act = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, rgb_x, dep_x):
        rgb_x = self.rgb_act(self.rgb_bn(self.rgb_conv(rgb_x)))
        dep_x = self.dep_act(self.dep_bn(self.dep_conv(dep_x)))

        x = torch.mul(rgb_x, dep_x)

        x = self.att_act(self.att_bn(self.att_conv(x)))
        x = self.out_act(self.out_bn(self.out_conv(x)))
        x = self.dropout(x)
        
        return x 

class DAttNet(nn.Module):
    def __init__(self, dim_size=512):
        super().__init__()
        self.rgb_enc = get_vgg16(in_channels=3)  
        self.dep_enc = get_vgg16(in_channels=1) 
        self.attn_layer = DepthAttention(in_ch=dim_size)

    def forward(self, x):
        # feature extraction
        rgb_x = self.rgb_enc(x[:,:3,:,:])
        dep_x = self.dep_enc(x[:,-1:,:])
        att_map = self.attn_layer(rgb_x, dep_x)
        return (rgb_x * att_map)
