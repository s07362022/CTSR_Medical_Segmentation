import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算欧氏距离的平方
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True).pow(2)

        # 计算对比损失
        contrastive_loss = torch.mean((1 - label) * euclidean_distance + 
                                      label * F.relu(self.margin - euclidean_distance))

        return contrastive_loss

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=2, alpha=2, gamma=3):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(predicted, target, num_classes=4, smooth=1e-5):
    smooth = 1e-5  # 平滑因子，用于避免分母为0

    # predicted = torch.softmax(predicted, dim=1)  # 对 predicted 进行softmax处理
    losses = []

    for class_index in range(num_classes):
        predicted_class = predicted[:, class_index]  # 取出对应类别的预测结果
        target_class = (target == class_index).float()  # 创建对应类别的目标张量

        predicted_class = predicted_class.view(predicted_class.size(0), -1)
        target_class = target_class.view(target_class.size(0), -1)

        intersection = torch.sum(predicted_class * target_class)
        union = torch.sum(predicted_class) + torch.sum(target_class)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice
        # if class_index ==3:
        # print('class_index{} loss:',format(class_index))
        # print(dice_loss)

        losses.append(dice_loss)

    loss = sum(losses) / num_classes  # 计算所有类别的平均 Dice Loss

    return loss
def Dice_loss2(predicted, target, beta=1, smooth = 1e-5):
    smooth = 1e-5  # 平滑因子，用于避免分母为0
    predicted = torch.softmax(predicted, dim=1)  # 对 predicted 进行softmax处理

    # 计算 Dice Loss
    
    predicted1 = predicted[:, 1]  # 取出第二类的预测结果
    predicted1 = predicted1.view(predicted1.size(0), -1)  # 将预测结果展平
    target = target.view(target.size(0), -1)  # 将目标标签展平

    intersection = torch.sum(predicted1 * target)
    union = torch.sum(predicted1) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss1 = 1.0 - dice
    
#     predicted2 = predicted[:, 0]  # 取出第1类的预测结果
#     predicted2 = predicted2.view(predicted2.size(0), -1)  # 将预测结果展平
#     target = target.view(target.size(0), -1)  # 将目标标签展平

#     intersection = torch.sum(predicted2 * target)
#     union = torch.sum(predicted2) + torch.sum(target)
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     loss2 = 1.0 - dice
    
    loss = loss1#+loss2
    
    return loss

def Dice_loss3(inputs, target, beta=2, smooth = 1e-5):
    inputs = torch.argmax(inputs, dim=1)
    n, c, h = inputs.size()
    
    nt, ht, wt= target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.view(n, -1, c),-1)
    # temp_inputs = torch.argmax(temp_inputs, dim=1)
    temp_target = target.view(n, -1, 1)
    print(temp_target.shape)
    print(temp_inputs.shape)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='xavier', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 计算focloss
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss2(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss2,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds = preds.float()
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
        print(labels.shape)
        print(preds_softmax.shape)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

    
import os
import numpy as np
from skimage import measure


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice



def dice_score_batch(prediction, target):
    smooth = 1e-5
    batchsize = target.size(0)
    num_classes = target.size(1)
    prediction = prediction.view(batchsize, num_classes, -1)
    target = target.view(batchsize, num_classes, -1)
    intersection = (prediction * target)
    dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
    return dice


def measure_img(o_img, t_num=1):
    p_img=np.zeros_like(o_img)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img