'''
在124批次数据上训练，3批次数据上测试
'''
import os
import torch
from torch.utils.data import DataLoader

from Dataloader.dataloader import \
    Train_Dloader, \
    SamplerByLabel, \
    TestRegionLevel_Dloader
from models.SINetV2 import ModelSINetV2
from utils import \
    save_weight, \
    load_weight, \
    PR_score, \
    BCE_criterion, \
    BCE_neg_pixellevel_criterion, \
    DICE_criterion


### 模型训练过程中的一些基本参数的设置 ###
class Opt:
    def __init__(self):
        self.gpu = 3  # 指定gpu
        self.cls = 1  # 预测的类别
        self.size = 320  # 图像块的裁切大小
        '''
        label
        =0采样Dimple图片,
        =1采样压痕不良图片,
        =2采样脏污等其他非Dimple图片,
        =3采样可能Dimple可能脏污的难例图片
        '''
        self.labels = 16 * [0] + 8 * [1] + 8 * [2] + 8 * [3]
        self.lr = 1e-3  # 学习率
        self.iterations = 2000  # 迭代次数
        self.root = '/home/wujianxiong/mywork/Dimple/Dimple-dataset/'  # 数据集根目录
        self.weight_pth = 'ModelSINetV2/{}.pth'


### 模型训练, 验证 + 保存 ###
def main(opt):
    #########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### init ###
    size = opt.size
    labels = opt.labels
    lr = opt.lr
    iterations = opt.iterations
    root = opt.root
    weight_pth = opt.weight_pth
    ############

    model = ModelSINetV2()
    model, step = load_weight(model, 'none.pth', device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    Train = Train_Dloader(root=root,
                          bbox_json='Dataloader/bboxes.json',
                          seg_root='Dataloader/seg_set/',
                          size=size)
    train_data = DataLoader(Train,
                            num_workers=6,
                            batch_size=len(labels),
                            collate_fn=Train.collate_fn,
                            sampler=SamplerByLabel(labels))
    val_data = TestRegionLevel_Dloader(root=root,
                                       bbox_json='Dataloader/bboxes.json',
                                       size=size)

    model.train()
    for flag, img, seg in train_data:
        flag = flag.to(device)
        img = img.to(device)
        seg = seg.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels).to(device)

        optimizer.zero_grad()
        logit, seg_logit = model(img)

        '''
        每个mini batch(batch size=40)中
        对于Dimple图片，16张图片忽略1张
        对于难例图片，默认为负例，但是忽略1/4的图片（2张）
        合计忽略3张图片
        '''
        loss_cls = BCE_criterion(logit[labels == 0], label=1, ignore_k=1) + \
                   BCE_criterion(logit[labels == 1], label=0, ignore_k=0) + \
                   BCE_criterion(logit[labels == 2], label=0, ignore_k=0) + \
                   BCE_criterion(logit[labels == 3], label=0, ignore_k=2)
        loss_cls = loss_cls / (len(labels) - 3)
        loss_seg = 0
        if flag.sum() != 0:
            loss_seg = DICE_criterion(seg_logit[flag == 1], seg)
        loss_seg_neg = BCE_neg_pixellevel_criterion(seg_logit[(labels == 1)+(labels == 2)])
        loss = loss_cls + loss_seg + 0.05 * loss_seg_neg

        loss.backward()
        optimizer.step()

        # 打印
        step += 1
        if step % 10 == 0:
            print('loss_cls = {:.4f}, loss_seg = {:.4f}, step = {}, iterations={}'
                  .format(loss_cls, loss_seg, step, iterations))

        # 验证+保存
        if step % 2000 == 0:
            probs = list()
            gts = list()
            model.eval()
            for img_pth, img, label in val_data():
                img = img.to(device)
                with torch.no_grad():
                    prob = model(img)[0].sigmoid().min()
                    prob = prob.data.cpu().tolist()
                probs.append(prob)
                gts.append(label)
                # 打印残差>0.3的所有样本
                if abs(label - prob) >= 0.3:
                    print('{}: pred prob={}, label={}'.format(
                        img_pth,
                        round(prob, 2),
                        label)
                    )
            model.train()
            acc, recall = PR_score(gts, probs)
            print('acc={:.4f}, recall={:.4f}'.format(acc, recall))
            save_weight(model, step, weight_pth.format(step))
        if step >= iterations:
            break


if __name__ == '__main__':
    opt = Opt()
    # opt.gpu = 0
    main(opt)
