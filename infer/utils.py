import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm
from skimage import measure
import cv2
import os

from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation._utils import _SimpleSegmentationModel


def red_map(ori_img, std_min=30):
    '''
    :param ori_img: BGR通道的原始图片
    :param std_min: 图像上每个pixel在色彩上的std范围, 在范围内的点保留
    :return: act_map, 可以理解为提取出来的图片中的红色框
    '''
    H, W, _ = ori_img.shape
    act_map = np.zeros((H, W), dtype='float32')
    std = np.std(ori_img, axis=-1)
    # 1. 添加红色框
    act_map[(std > std_min) *
            (ori_img[:, :, 2] == ori_img[:, :].max(axis=-1))] = 1
    return act_map


def pooling(att_map, kernel_size=149, stride=8, threshold=0.02):
    '''
    :param att_map: 之前提取的红色框, [0, 1], shape = HW
    :param kernel_size: average pooling size
    :param stride: 下采样步长，减小计算量
    :param threshold: 激活值的threshold
    :return: 一张mask, mask中每个孤立的区域为原图中有红框的大致区域
    '''
    H, W = att_map.shape
    att_map = torch.from_numpy(att_map[np.newaxis, np.newaxis]).float()
    if torch.cuda.is_available():
        att_map = F.avg_pool2d(att_map.cuda(), kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        att_map = att_map[0, 0].cpu().numpy()
    else:
        att_map = F.avg_pool2d(att_map, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        att_map = att_map[0, 0].numpy()
    att_map[att_map >= threshold] = 1
    att_map[att_map != 1] = 0
    att_map = cv2.resize(att_map.astype('uint8'), (H, W), interpolation=cv2.INTER_NEAREST)
    return att_map


######## model ##########
class Resnet18_(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(Resnet18_, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        del self.resnet.fc
        self.fc = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

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


class Resnet18(Resnet18_):
    def __init__(self, *args, **kwargs):
        super(Resnet18, self).__init__(*args, **kwargs)
        self.scSE1 = scSE_Module(channel=64, ratio=16)
        self.scSE2 = scSE_Module(channel=128, ratio=16)
        self.scSE3 = scSE_Module(channel=256, ratio=32)
        self.scSE4 = scSE_Module(channel=512, ratio=32)

        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fc = nn.Linear(21 * 256, 1)

        '''
        修改了下采样的步长，那就修改卷积的空洞率，以保证修改后的模型和
        修改前的模型感受野大小一致
        '''
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

        # 以下为目标检测所用的SPP与其实现
        x = self.conv1x1(x)
        x = torch.cat([
            F.adaptive_avg_pool2d(x, (4, 4)).flatten(1),
            F.adaptive_avg_pool2d(x, (2, 2)).flatten(1),
            F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)],
            dim=-1)
        x = self.fc(x)
        ############################
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
#########################


def get_bboxes(att_map, min_size=500):
    # 连通域标记
    label_image = measure.label(att_map)
    boundingbox = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        if region.area >= min_size:
            boundingbox.append(region.bbox)
    return boundingbox


def load_weight(net, net_pth, device):
    step = 0
    if os.path.isfile(net_pth):
        save_dic = torch.load((net_pth), map_location=lambda storage, loc: storage)
        net.load_state_dict(save_dic['state_dict'])
        step = save_dic['step']
        print('weight loaded successfully')
    else:
        print('no weight file')
    net = net.to(device)
    return net, step


def edge_det(ori_img, std_min=4):
    '''
    :param ori_img: BGR通道的原始图
    '''
    H, W = ori_img.shape[:2]
    act_map = np.zeros((H, W), dtype='uint8')
    std = np.std(ori_img, axis=-1)
    mean = np.mean(ori_img, axis=-1)
    # 1. 只保留色彩通道上方差大于std_min的点
    act_map[std > std_min] = 1
    return act_map


def guideMedianFilter(img, guide_map, ksize=9):
    dst = cv2.medianBlur(img, ksize=ksize)
    img, guide_map, dst = img.astype('int16'), guide_map.astype('int16'), dst.astype('int16')
    img = img * (1 - guide_map) + dst * guide_map
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def img_norm(img):
    img = img.astype('float32')
    return (img - img.mean()) / (img.std() + 1e-5)


def main(opt):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    ## init ##
    weight = opt.weight
    img_pths = opt.img_pths
    size = opt.size
    max_bbox_num = opt.max_bbox_num
    ##########

    model = FCN_res18(num_classes=1)
    model, _ = load_weight(model, weight, device)
    model.eval()

    dic = dict()
    for pth in tqdm(img_pths):
        # get red bboxes
        img = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), -1)
        bboxes = get_bboxes(red_map(img), min_size=20)
        if len(bboxes) > max_bbox_num:
            prob = 0.0
        elif len(bboxes) == 0:
            prob = 0.0
        else:
            # filter
            act_map = edge_det(img)
            act_map = cv2.dilate(act_map, kernel=np.ones((2, 2)), iterations=1)  # dilate
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = guideMedianFilter(img, act_map, ksize=9)

            # img = cv2.imdecode(np.fromfile(pth.replace('grp', 'org'), dtype=np.uint8), 0)

            # crop
            H, W = img.shape
            img = img_norm(img)
            img = np.pad(img,
                         ((size // 2, size - size // 2),
                          (size // 2, size - size // 2)),
                         'constant',
                         constant_values=0,
                         )
            img_patch = list()
            for bbox in bboxes:
                center = [np.clip((bbox[0] + bbox[2]) // 2, size // 2, H - size + size // 2),
                          np.clip((bbox[1] + bbox[3]) // 2, size // 2, W - size + size // 2)]
                img_patch.append(img[center[0]: center[0] + size, center[1]: center[1] + size])
            img_patch = np.stack(img_patch, axis=0)[:, np.newaxis]
            img_patch = torch.from_numpy(img_patch).float().to(device)
            with torch.no_grad():
                prob = model(img_patch)[0].sigmoid().min()
                prob = prob.data.cpu().tolist()
        print('{} 预测Dimple的概率: {:.4f}'.format(pth, prob))
        dic[pth] = prob
    return dic


def segment(opt):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    ## init ##
    weight = opt.weight
    img_pths = opt.img_pths
    size = opt.size
    max_bbox_num = opt.max_bbox_num
    ##########

    model = FCN_res18(num_classes=1)
    model, _ = load_weight(model, weight, device)
    model.eval()

    dic = dict()
    for pth in tqdm(img_pths):
        # filter
        act_map = edge_det(img)
        act_map = cv2.dilate(act_map, kernel=np.ones((2, 2)), iterations=1)  # dilate
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = guideMedianFilter(img, act_map, ksize=9)

        # img = cv2.imdecode(np.fromfile(pth.replace('grp', 'org'), dtype=np.uint8), 0)

        img_patch = img_norm(img)
        img_patch = img_patch[np.newaxis, np.newaxis]
        img_patch = torch.from_numpy(img_patch).float().to(device)
        with torch.no_grad():
            pred = model(img_patch)[-1].sigmoid()
            pred = pred[0].cpu().numpy()
            pred[pred <= 0.5] = 0

        # visual
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pred = (pred * 255).astype('uint8').transpose(1, 2, 0).repeat(3, -1)
        pred[:, :, :2] = 0
        cv2.addWeighted(pred, 0.3, img, 0.7, 0, img)
        yield pth, img
