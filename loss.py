#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2020/04/10
@Author  :   Beta_Tao
@Version :   1.0
@Contact :   wangtao_0902@163.com
@Desc    :   Calculate the loss value
'''

# TODO: 重新看一遍
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.autograd import Function
import torch.nn.functional as F

from bbox import encodeBox

__all__ = ['EzDetectLoss']


# TODO: buildboxTarget 全部看不懂
def buildbboxTarget(config, bboxOut, target):
    bboxMasks = torch.ByteTensor(bboxOut.size())
    bboxMasks.zero_()

    bboxTarget = torch.FloatTensor(bboxOut.size())

    # TODO: target类型以及存储的变量形式
    batchSize = target.size()[0]
    for i in range(0, batchSize):
        num = int(target[i][0])
        for j in range(0, num):
            offset = j * 6
            # cls = int(target[i][offset + 1])
            k = int(target[i][offset + 6])
            trueBox = [target[i][offset + 2], target[i][offset + 3], target[i][offset + 4], target[i][offset + 5]]
            predBox = config.predBoxes[k]
            ebox = encodeBox(config, trueBox, predBox)
            bboxMasks[i, k, :] = 1
            bboxTarget[i, k, 0] = ebox[0]
            bboxTarget[i, k, 1] = ebox[1]
            bboxTarget[i, k, 2] = ebox[2]
            bboxTarget[i, k, 3] = ebox[3]

    # if config.gpu:
    #     bboxMasks = bboxMasks.cuda()
    #     bboxTartget = bboxTarget.cuda()

    return bboxMasks, bboxTarget


# TODO: buildConfTarget 全部看不懂
def buildConfTarget(config, confOut, target):
    batchSize = confOut.size()[0]
    boxNumber = confOut.size()[1]

    confTarget = torch.LongTensor(batchSize, boxNumber, config.classNumber)

    confMasks = torch.ByteTensor(confOut.size())
    confMasks.zero_()

    confScore = F.log_softmax(Variable(confOut.view(-1, config.classNumber), requires_grad=False))
    confScore = confScore.data.view(batchSize, boxNumber, config.classNumber)

    # positive samples
    pnum = 0
    for i in range(0, batchSize):
        num = int(target[i][0])
        for j in range(0, num):
            offset = j * 6
            k = int(target[i][offset + 6])
            cls = int(target[i][offset + 1])
            if cls > 0:
                confMasks[i, k, :] = 1
                confTarget[i, k, :] = cls
                confScore[i, k, :] = 0
                pnum += 1
            else:
                confScore[i, k, :] = 0

    # negtive samples (background)
    confScore = confScore.view(-1, config.classNumber)
    confScore = confScore[:, 0].contiguous().view(-1)

    scoreValue, scoreIndex = torch.sort(confScore, 0, descending=False)

    # 由于正负样本差别巨大， 所以需要根据负样本的置信度进行排序并取整体正负比例为1：3
    for i in range(pnum * 3):
        b = scoreIndex[i] // boxNumber
        k = scoreIndex[i] % boxNumber
        if confMasks[b, k, 0] > 0:
            break
        confMasks[b, k, :] = 1
        confTarget[b, k, :] = 0

    # if config.gpu:
    #     confMasks = confMasks.cuda()
    #     confTarget = confTarget.cuda()

    return confMasks, confTarget


class EzDetectLoss(nn.Module):
    def __init__(self, config, pretrained=False):
        super(EzDetectLoss, self).__init__()
        self.config = config
        self.confLoss = nn.CrossEntropyLoss()
        self.bboxLoss = nn.SmoothL1Loss()

    def forward(self, confOut, bboxOut, target):
        # batchSize = target.size()[0]

        # building loss of conf
        confMasks, confTarget = buildConfTarget(self.config, confOut.data, target)
        confSamples = confOut[confMasks].view(-1, self.config.classNumber)
        confTarget = confTarget[confMasks].view(-1, self.config.classNumber)
        confTarget = confTarget[:, 0].contiguous().viwe(-1)
        confTarget = Variable(confTarget, requires_grad=False)

        confLoss = self.confLoss(confSamples, confTarget)

        # building loss of bbox
        bboxMasks, bboxTarget = buildbboxTarget(self.config, bboxOut.data, target)

        bboxSamples = bboxOut[bboxMasks].view(-1, 4)
        bboxTarget = bboxTarget[bboxMasks].view(-1, 4)
        bboxTarget = Variable(bboxTarget)
        bboxLoss = self.bboxLoss(bboxSamples, bboxTarget)

        return confLoss, bboxLoss
