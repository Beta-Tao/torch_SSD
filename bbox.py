import math

import torch

__all__ = ['bboxIOU', 'encodeBox', 'decodeAllBox', 'doNMS']


def bboxIOU(bboxA: list, bboxB: list):
    ''' Calculate the IOU value between two boxes.
    
    Args:
        bboxA, bboxB(list): The box information.
    
    Returns:
        IOU(int): The IOU value
    
    Raises:
        None
    '''
    
    A_xmin = bboxA[0]
    A_ymin = bboxA[1]
    A_xmax = bboxA[2]
    A_ymax = bboxA[3]
    A_width = A_xmax - A_xmin
    A_height = A_ymax - A_ymin

    B_xmin = bboxB[0]
    B_ymin = bboxB[1]
    B_xmax = bboxB[2]
    B_ymax = bboxB[3]
    B_width = B_xmax - B_xmin
    B_height = B_ymax - B_ymin

    xmin = min(A_xmin, B_xmin)
    ymin = min(A_ymin, B_ymin)
    xmax = max(A_xmax, B_xmax)
    ymax = max(A_ymax, B_ymax)

    A_width_and = (A_width + B_width) - (xmax - xmin)
    A_height_and = (A_height + B_height) - (ymax - ymin)

    if A_width_and <= 0.0001 or A_height_and <= 0.0001:     # 没有交集
        return 0

    area_and = A_width_and * A_height_and
    area_or = (A_width * A_height) + (B_width * B_height) - area_and
    IOU = area_and / area_or

    return IOU


def encodeBox(config, box, predBox):
    ''' Calculate the error between bbox with predBox, notice that there are multiple transformations.
            The function force on a pair of box and predBox
    
    Args:
        config: EzDetectConfig
        box: The bounding box
        predBox : The predict box
    
    Returns:
        The error of box size(list): [ecx, ecy, ew, eh]
    
    Raises:
        None
    '''

    pcx = (predBox[0] + predBox[2]) / 2
    pcy = (predBox[1] + predBox[3]) / 2
    pw = predBox[2] - predBox[0]
    ph = predBox[3] - predBox[1]

    # 中心的相对偏差 乘10倍
    ecx = (box[0] + box[2]) / 2 - pcx
    ecy = (box[1] + box[3]) / 2 - pcy
    ecx = ecx / pw * 10
    ecy = ecy / ph * 10

    # 尺寸的偏差
    ew = (box[2] - box[0]) / pw
    eh = (box[3] - box[1]) / ph
    ew = math.log(ew) * 5
    eh = math.log(eh) * 5

    return [ecx, ecy, ew, eh]


def decodeAllBox(config, allBox):
    ''' Calculate the bounding boxes from the error of box size and predBox.
            The function handle the errorBox in tensor
    
    Args:
        config: EzDetectConfig.
        allBox(tensor, [batchIndex][predBoxIndex][ecx, ecy, ew, eh]): Three dimensional tensor.
    
    Returns:
        newBoxes(tensor, [batchIndex][predBoxIndex][bbox[0], bbox[1], bbox[2], bbox[3]]):
            Three dimensional tensor.

    Raises:
        None
    '''
    
    newBoxes = torch.FloatTensor(allBox.size())
    batchSize = newBoxes.size()[0]

    for k in range(len(config.predBoxes)):
        predBox = config.predBoxes[k]
        pcx = (predBox[0] + predBox[2]) / 2
        pcy = (predBox[1] + predBox[3]) / 2
        pw = predBox[2] - predBox[0]
        ph = predBox[3] - predBox[1]

        for i in range(batchSize):
            box = allBox[i, k, :]

            dcx = box[0] / 10 * pw + pcx
            dcy = box[1] / 10 * ph + pcy
            dw = math.exp(box[2] / 5) * pw
            dh = math.exp(box[3] / 5) * ph

            newBoxes[i, k, 0] = max(0, dcx - dw / 2)
            newBoxes[i, k, 1] = max(0, dcy - dh / 2)
            newBoxes[i, k, 2] = min(1, dcx + dw / 2)
            newBoxes[i, k, 3] = min(1, dcy + dh / 2)
    
    # if config.gpu:
    #     newBoxes = newBoxes.cuda()

    return newBoxes


def doNMS(config, classMap, allBoxes, threshold):
    ''' Use Non-MaximumSuppression(NMS) to merge the result boxes.
        The principle of NMS is to get the score of all boxes and save
        the boxe with bigger score if two boxes's IOU is beyond threshold.
    
    Args:
        config: EzDetectConfig
        classMap(list):
        allBoxes:
        threshold: The threshold which deceides whether to merge.
    
    Returns:
        ret_val
    
    Raises:
        None
    '''

    winBoxes = []
    # predBoxes = config.predBoxes

    for c in range(1, config.classNumber):
        fscore = classMap[:, c]
        v, s = torch.sort(fscore, 0, descending=True)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>', c, v[0])
        for i in range(len(v)):
            if v[i] < threshold:
                continue
            k = s[i]
            boxA = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]
            for j in range(i + 1, len(v)):
                if v[j] < threshold:
                    continue

                k = s[j]
                boxB = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]

                iouValue = bboxIOU(boxA, boxB)
                if iouValue > 0.5:
                    v[j] = 0

        for i in range(len(v)):
            if v[j] < threshold:
                continue

            k = s[i]
            box = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]

            winBoxes.append(box)
    return winBoxes
