#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sampling.py
@Time    :   2020/04/10
@Author  :   Beta_Tao
@Version :   1.0
@Contact :   wangtao_0902@163.com
@Desc    :   None
'''

# import random
# import math

import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.autograd import Function

from bbox import bboxIOU

__all__ = ['buildPredBoxes', 'sampleEzDetect']


def buildPredBoxes(config):
    ''' Build predict boxes according to the size of anchors
        and feature layers.
    
    Args:
        config: The EzDetectConfig.
    
    Returns:
        A list store the relative coordinates of all boxes.
        [predboxIndex][xmin, ymin, xmax, ymax]
    
    Raises:
        None
    '''
    # 存储5个特征层中所有对应尺寸的anchor box
    predBoxes = []

    for i in range(len(config.mboxes)):
        # 特征层索引
        layerIdx = config.mboxes[i][0]

        # 取出此特征层的输出尺寸
        wid = config.featureSize[layerIdx][0]
        hei = config.featureSize[layerIdx][1]

        # 取出anchor的尺寸
        wbox = config.mboxes[i][1]
        hbox = config.mboxes[i][2]

        for y in range(hei):
            for x in range(wid):
                # x, y位置取每个特征层像素中心点来计算
                xc = (x + 0.5) / wid
                yc = (y + 0.5) / hei

                # 得到box在特征层上的比例
                xmin = xc - wbox / 2
                ymin = yc - hbox / 2
                xmax = xc + wbox / 2
                ymax = yc + hbox / 2

                predBoxes.append([xmin, ymin, xmax, ymax])

    return predBoxes


def sampleEzDetect(config, bboxes: list):
    ''' Sample the bboxes and predIndex according to IOU value
    
    Args:
        config: EzDetectConfig
        bboxes: The bboxes solved from the xml file
    
    Returns:
        selectedSamples(list): Stored the information with high IOU
        [0] the sample number
        [index * 6 + 1] the positive or negative name of className
        [index * 6 + 2 to 5] the corresponding bboxe's relative coordinates
        [index * 6 + 6] the corresponding predBox index
    
    Raises:
        errorType
    '''

    predBoxes = config.predBoxes

    # preparing groud truth
    truthBoxes = []

    for i in range(len(bboxes)):
        # 获取正确的bbox坐标
        truthBoxes.append([bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]])

    # computing iou
    iouMatrix = []
    for i in predBoxes:
        # 存储该predBoxes对应的所有truthBox的IOU
        ious = []
        for j in truthBoxes:
            ious.append(bboxIOU(i, j))
        iouMatrix.append(ious)

    iouMatrix = torch.FloatTensor(iouMatrix)
    iouMatrix2 = iouMatrix.clone()

    ii = 0
    selectedSamples = torch.FloatTensor(128 * 1024)

    # positive samples from bi-direction match
    # 最后selectedSamples中为所有bboxes对应的iou最大的predIndex以及位置信息
    for i in range(len(bboxes)):
        iouViewer = iouMatrix.view(-1)  # 变成一维的结构
        # 找到iou最大值的值和索引
        iouValues, iouSequence = torch.max(iouViewer, 0)

        # TODO: iouSequence[0]中存着什么元素？ 这里可能运行时会报错，添加item()
        # 由于上面将iouMatrix转换为一维，所以这里使用取下整除以及余数得到predIndex以及bboxIndex
        predIndex = iouSequence[0] // len(bboxes)
        bboxIndex = iouSequence[0] % len(bboxes)

        if iouValues[0] > 0.1:
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex

            ii += 1
        else:
            break

        # 清除iou较大的值，承接后面的循环
        iouMatrix[:, bboxIndex] = -1
        iouMatrix[predIndex, :] = -1
        iouMatrix2[predIndex, :] = -1

    # also samples with high iou
    for i in range(len(predBoxes)):
        # 注意此时最大的iou对应的iouMatrix2列已经变成-1
        v, _ = iouMatrix2[i].max(0)

        predIndex = i
        bboxIndex = _[0]

        if v[0] > 0.7:
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex

            ii += 1
        elif v[0] < 0.5:
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0] * -1
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex

            ii += 1

    selectedSamples[0] = ii
    return selectedSamples
