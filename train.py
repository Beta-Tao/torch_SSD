from __future__ import print_function
import argparse

# from math import log10

# import torch
# import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset import vocDataset as DataSet
from model import EzDetectNet
from model import EzDetectConfig
from loss import EzDetectLoss

parser = argparse.ArgumentParser(description='EasyDetect by pytorch')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1024, help='random seed to use. Default=123')
parser.add_argument('--gpu', dest='gpu', action='store_false')

parser.set_defaults(gpu=False)
opt = parser.parse_args()

print('===> Loading datasets')
ezConfig = EzDetectConfig(opt.batchSize, opt.gpu)
train_set = DataSet(ezConfig, True)
test_set = DataSet(ezConfig, False)

train_data_loader = DataLoader(dataset=train_set,
                               num_workers=opt.threads,
                               batch_size=opt.batchSize,
                               shuffle=True)
test_data_loader = DataLoader(dataset=test_set,
                              num_workers=opt.threads,
                              batch_size=opt.batchSize)

print('===> Building model')
mymodel = EzDetectNet(ezConfig, True)
myloss = EzDetectLoss(ezConfig)
optimizer = optim.SGD(mymodel.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

if ezConfig.gpu is True:
    mymodel.cuda()
    myloss.cuda()

# TODO: The rest
