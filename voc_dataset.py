#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   voc_dataset.py
@Time    :   2020/04/10
@Author  :   Beta_Tao
@Version :   1.0
@Contact :   wangtao_0902@163.com
@Desc    :   Load VOC2007 data
'''

from os import listdir
from os.path import join
from random import random
from PIL import Image
import xml.etree.ElementTree as ET

# import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect

# 给出对外的接口
__all__ = ['vocClassName', 'vocClassID', 'vocDataset']

vocClassName = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor']


def getVOCInfo(xmlFile: str):
    ''' Get class name and boxes from a xml file
    
    Args:
        xmlFile: A xml file path
    
    Returns:
        A list that stores information about bounding boxes in an image.
        The x, y with min & max is the absolute coordinates in a image
        [bboxIndex]['category_id']: class name
        [bboxIndex]['bbox'][0]: xmin
        [bboxIndex]['bbox'][1]: ymin
        [bboxIndex]['bbox'][2]: xmax
        [bboxIndex]['bbox'][3]: ymax
    
    Raises:
        None
    '''

    root = ET.parse(xmlFile).getroot()
    anns = root.findall('object')
    bboxes = []

    for ann in anns:
        newAnn = {}

        name = ann.find('name').text
        newAnn['category_id'] = name

        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1, -1, -1, -1]
        newAnn['bbox'][0] = float(bbox.find('xmin').text)
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)

        bboxes.append(newAnn)
    return bboxes


class vocDataset(data.Dataset):
    ''' Store and pretreatment VOC data
    
    Attributes:
        config: The EzDetectConfig class
        isTraining: Mark is the dataset training or testing
        transformer: The function with transforme method for image
    '''
    
    def __init__(self, config, isTraining=True):
        super(vocDataset, self).__init__()
        self.isTraining = isTraining
        self.config = config

        # 利用均值和方差对图片中的RGB值进行归一化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([transforms.ToTensor(), normalize])

    # 针对数据进行预处理
    def __getitem__(self, index):
        item = None

        if self.isTraining:
            item = allTrainingData[index % len(allTrainingData)]
        else:
            item = allTestingData[index % len(allTestingData)]

        img = Image.open(item[0])   # 图像数据
        imgWidth, imgHeight = img.size

        allBboxes = getVOCInfo(item[1])     # item[1]为对应的xml内容

        # 对图片进行随机crop，并保证bbox的大小
        # TODO: 随机crop原图片的原因
        targetWidth = int((random() * 0.25 + 0.75) * imgWidth)
        targetHeight = int((random() * 0.25 + 0.75) * imgHeight)
        xmin = int(random() * (imgWidth - targetWidth))
        ymin = int(random() * (imgHeight - targetHeight))
        img = img.crop((xmin, ymin, xmin + targetWidth, ymin + targetHeight))
        img = img.resize((self.config.targetWidth, self.config.targetHeight), Image.BILINEAR)
        imgT = self.transformer(img)
        imgT = imgT * 256

        # 调整bbox
        # TODO: 调整bbox的原理未懂
        bboxes = []
        for i in allBboxes:
            xl = i['bbox'][0] - xmin
            yt = i['bbox'][1] - ymin
            xr = i['bbox'][2] - xmin
            yb = i['bbox'][3] - ymin
            if xl < 0:
                xl = 0
            if xr >= targetWidth:
                xr = targetWidth - 1
            if yt < 0:
                yt = 0
            if yb >= targetHeight:
                yb = targetHeight - 1
            xl = xl / targetWidth
            xr = xr / targetWidth
            yt = yt / targetHeight
            yb = yb / targetHeight

            if (xr - xl >= 0.05 and yb - yt >= 0.05):
                bbox = [vocClassID[i['category_id']], xl, yt, xr, yb]
                bboxes.append(bbox)

        if len(bboxes) == 0:
            return self[index + 1]

        target = sampleEzDetect(self.config, bboxes)
        return imgT, target

    def __len__(self):
        # 整数倍batchSize的长度
        if self.isTraining:
            num = len(allTrainingData) - (len(allTrainingData) % self.config.batchSize)
            return num
        else:
            num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
            return num


vocClassID = {}
# 构造类别名称对应的id，id从1开始
for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i + 1
print(vocClassID)

allTrainingData = []
allTestingData = []

allFolder = ['/home/wangtao/pythonTest/VOCdevkit/VOC2007']

for folder in allFolder:
    imagePath = join(folder, 'JPEGImages')
    infoPath = join(folder, 'Annotations')
    index = 0
    for f in listdir(imagePath):    # 遍历9964张原始图片
        if f.endswith('.jpg'):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4] + '.xml')
            if index % 10 == 0:  # 每10张随机抽1个样本进行测试
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.append((imageFile, infoFile))

            index += 1
