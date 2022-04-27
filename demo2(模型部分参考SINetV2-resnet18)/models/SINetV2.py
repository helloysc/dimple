import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

'''
scSE block
'''
class cSE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(cSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel, ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


'''
resnet18 encoder
'''
class Resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        del self.resnet.fc

        self.scSE1 = scSE_Module(channel=64, ratio=16)
        self.scSE2 = scSE_Module(channel=128, ratio=16)
        self.scSE3 = scSE_Module(channel=256, ratio=32)
        self.scSE4 = scSE_Module(channel=512, ratio=32)

    def forward(self, x):
        features = []
        x = x.repeat(1, 3, 1, 1)  # N1HW -> N3HW
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.scSE1(x)
        x = self.resnet.layer2(x)
        x = self.scSE2(x)
        features.append(x)
        x = self.resnet.layer3(x)
        x = self.scSE3(x)
        features.append(x)
        x = self.resnet.layer4(x)
        x = self.scSE4(x)
        features.append(x)

        return features

'''
TEM block(from SINet V2)
'''
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TEM_module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TEM_module, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


'''
NCD module(from SINet V2)
'''
class NCD_module(nn.Module):
    def __init__(self, channel=32):
        super(NCD_module, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


'''
Model
'''
class ModelSINetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18_encoder = Resnet18()

        self.TEM2 = TEM_module(128, 32)
        self.TEM3 = TEM_module(256, 32)
        self.TEM4 = TEM_module(512, 32)

        self.NCD = NCD_module()

        self.fc = nn.Linear(35, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x2, x3, x4 = self.resnet18_encoder(x)

        x2 = self.TEM2(x2)
        x3 = self.TEM3(x3)
        x4 = self.TEM4(x4)

        feature_map = self.NCD(x4, x3, x2)

        x = torch.cat([
            F.adaptive_avg_pool2d(feature_map, (5, 5)).flatten(1),
            F.adaptive_avg_pool2d(feature_map, (3, 3)).flatten(1),
            F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)],
            dim=-1)

        x = self.fc(x)

        return x, F.interpolate(feature_map, size=input_shape, mode='bilinear', align_corners=False)
