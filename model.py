#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/04/10
@Author  :   Beta_Tao
@Version :   1.0
@Contact :   wangtao_0902@163.com
@Desc    :   Create SSD model class
'''

# import os
import math

import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torch.autograd import Function
# import torch.nn.functional as F
import torchvision.models as models

from sampling import buildPredBoxes

__all__ = ['EzDetectConfig', 'EzDetectNet']


class EzDetectConfig(object):
    ''' Store the detect information in net
    
    Attributes:
        batchSize(int): The number of images of the same batch.
        gpu(bool): Is open the GPU acceleration flag.
        classNumber(int): The class number need to classify.
        targetWidth(int): The input image width.
        targetHeight(int): The input image height.
        featureSize(list [layerIndex][width, height]): The size of five feature maps.
        mboxes(list [anchorIndex][layerIndex, anchorWidth, anchorHeight]): Anchor sizes on all feature maps.
        predBoxes(list [xmin, xmax, ymin, ymax]): Relative coordinates of anchors on all feature maps.
    '''
    
    def __init__(self, batchSize=4, gpu=False):
        super(EzDetectConfig, self).__init__()
        self.batchSize = batchSize
        self.gpu = gpu

        # 分类类别数量
        self.classNumber = 21

        # 输入图片的大小
        self.targetWidth = 330
        self.targetHeight = 330

        # 特征层大小
        self.featureSize = [[42, 42],   # L2 1/8
                            [21, 21],   # L3 1/16
                            [11, 11],   # L4 1/32
                            [6, 6],     # L5 1/64
                            [3, 3]]     # L6 1/110

        # 每个特征层anchor尺寸比例
        # TODO: mboxes中元素[2:]的意义
        priorConfig = [[0.10, 0.25, 2],
                       [0.25, 0.40, 2, 3],
                       [0.40, 0.55, 2, 3],
                       [0.55, 0.70, 2, 3],
                       [0.70, 0.85, 2]]

        self.mboxes = []    # 包含了所有的anchor信息 [0]为特征层索引 [1]为anchor宽方向 [2]为anchor高方向

        for i in range(len(priorConfig)):
            minSize = priorConfig[i][0]
            maxSize = priorConfig[i][1]
            meanSize = math.sqrt(minSize * maxSize)
            ratios = priorConfig[i][2:]

            # aspect ratio 1 for min and max
            self.mboxes.append([i, minSize, maxSize])
            self.mboxes.append([i, meanSize, meanSize])

            # other aspect ratio
            for r in ratios:
                ar = math.sqrt(r)
                self.mboxes.append([i, minSize * ar, minSize / ar])
                self.mboxes.append([i, minSize / ar, minSize * ar])

        # 根据不同特征层的anchor参数生成所有的anchors
        self.predBoxes = buildPredBoxes(self)


class EzDetectNet(nn.Module):
    ''' The information and config of SSD net module
    
    Attributes:
        config(EzDetectConfig): The EzDetectConfig.
        conv1: resnet.conv1
        bn1: resnet.bn1
        relu: resnet.relu
        maxpool: resnet.maxpool
        layer1: resnet.layer1
        layer2: resnet.layer2   The first feature map.
        layer3: resnet.layer3   The second feature map.
        layer4: resnet.layer4   The third feature map.
        layer5(module): The fourth feature map.
        layer6(module): The fifth feature map.
        confConvs(list [anchorIndex]):
            The convolution layer that output class info corresponding to each anchor.
        locConvs(list [anchorIndex]):
            The convolution layer that output bbox corrrdinates corresponding to each anchor.
    '''
    
    def __init__(self, config: EzDetectConfig, pretrained=False):
        super(EzDetectNet, self).__init__()
        self.config = config

        resnet = models.resnet50(pretrained)    # 从Pytorch的预训练库中获取ResNet50模型，直接载入
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        # 之后五层输出的特征层需要放置anchors
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # 第5层开始自定义
        self.layer5 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))

        # 对应五个特征层的输入尺寸
        inChannles = [512, 1024, 2048, 1024, 512]

        self.locConvs = []
        self.confConvs = []

        # 一个anchor对应两个head
        for i in range(len(config.mboxes)):
            inSize = inChannles[config.mboxes[i][0]]
            # 输出类别信息
            confConv = nn.Conv2d(inSize, config.classNumber, kernel_size=3, stride=1, padding=1, bias=True)
            # 输出坐标信息
            locConv = nn.Conv2d(inSize, 4, kernel_size=3, stride=1, padding=1, bias=True)
            self.locConvs.append(locConv)
            self.confConvs.append(confConv)
            super(EzDetectNet, self).add_module('{}_conf'.format(i), confConv)
            super(EzDetectNet, self).add_module('{}_loc'.format(i), locConv)

    def forward(self, x):
        ''' The forward of net module.
        
        Args:
            x: Input
        
        Returns:
            (locResult, confResult)
        
        Raises:
            None
        '''

        batchSize = x.size()[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # 五个特征层
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        featureSource = [l2, l3, l4, l5, l6]

        confs = []
        locs = []

        for i in range(len(self.config.mboxes)):
            x = featureSource[self.config.mboxes[i][0]]

            # 每个anchor都进入卷据层输出loc和conf
            loc = self.locConvs[i](x)
            loc = loc.permute(0, 2, 3, 1)   # 更换维度
            loc = loc.contiguous()  # 把tensor变成在内存上连续分布
            loc = loc.view(batchSize, -1, 4)    # 更换尺寸
            locs.append(loc)

            conf = self.confConvs[i](x)
            conf = conf.permute(0, 2, 3, 1)
            conf = conf.contiguous()
            conf = conf.view(batchSize, -1, self.config.classNumber)
            confs.append(conf)

        # 按列拼接
        locResult = torch.cat(locs, 1)
        confResult = torch.cat(confs, 1)
        return locResult, confResult
