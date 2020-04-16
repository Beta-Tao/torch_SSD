import torch

iouMatrix = torch.tensor([[-0.3623, -0.6115],
                          [0.7283, 0.4699],
                          [2.3261, 0.1599]])

iouViewer = iouMatrix.view(-1)  # 变成一维的结构
print(iouViewer)
# 找到iou最大值的值和索引
iouValues, iouSequence = torch.max(iouViewer, 0)

print(iouValues)
print(iouSequence)
# print(iouSequence[0])
print(iouMatrix[0])
