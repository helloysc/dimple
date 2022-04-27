'''
FCN基于resnet18,
加入了scSE block,
分类头采用了SPP head
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation._utils import _SimpleSegmentationModel

from .resnet18 import Resnet18


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


class Resnet18(Resnet18):
    def __init__(self, *args, **kwargs):
        super(Resnet18, self).__init__(*args, **kwargs)
        self.scSE1 = scSE_Module(channel=64, ratio=16)
        self.scSE2 = scSE_Module(channel=128, ratio=16)
        self.scSE3 = scSE_Module(channel=256, ratio=32)
        self.scSE4 = scSE_Module(channel=512, ratio=32)

        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fc = nn.Linear(21 * 256, 1)

        for m in self.resnet.layer3.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                if isinstance(m.stride, int):
                    m.stride = (m.stride, m.stride)
                if m.stride[0] == 2:
                    m.stride = (1, 1)
                elif isinstance(m, nn.Conv2d):
                    m.dilation = 2
                    m.padding = m.dilation
        for m in self.resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                if isinstance(m.stride, int):
                    m.stride = (m.stride, m.stride)
                if m.stride[0] == 2:
                    m.stride = (1, 1)
                elif isinstance(m, nn.Conv2d):
                    m.dilation = 2
                    m.padding = m.dilation

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # N1HW -> N3HW
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.scSE1(x)
        x = self.resnet.layer2(x)
        x = self.scSE2(x)
        x = self.resnet.layer3(x)
        x = self.scSE3(x)
        x = self.resnet.layer4(x)
        x = self.scSE4(x)
        features = x
        x = self.conv1x1(x)

        ### 以下为目标检测所用的SPP与其实现 ###
        x = torch.cat([
            F.adaptive_avg_pool2d(x, (4, 4)).flatten(1),
            F.adaptive_avg_pool2d(x, (2, 2)).flatten(1),
            F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)],
            dim=-1)
        x = self.fc(x)
        #################################
        return features, x


class FCN_res18(_SimpleSegmentationModel):
    def __init__(self, num_classes=1):
        super(FCN_res18, self).__init__(
            backbone=Resnet18(num_classes=num_classes),
            classifier=FCNHead(in_channels=512, channels=num_classes),
            aux_classifier=None,
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features, cls_logits = self.backbone(x)
        seg_logits = self.classifier(features)
        seg_logits = F.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)
        return cls_logits, seg_logits
