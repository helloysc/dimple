import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Resnet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(Resnet18, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        del self.resnet.fc
        self.fc = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # N1HW -> N3HW
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        self.features = x

        x = self.resnet.avgpool(x)
        x = self.fc(x)
        return torch.flatten(x, 1)

    def infer(self, x, kernal_size=13):
        H, W = x.size(2), x.size(3)
        align_corners = True if kernal_size % 2 == 0 else False
        self.eval()
        with torch.no_grad():
            self.forward(x)
            x = F.avg_pool2d(self.features,
                             kernel_size=kernal_size,
                             stride=1,
                             padding=kernal_size // 2,
                             )
            x = self.fc(x)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=align_corners)
        return torch.sigmoid(x)
