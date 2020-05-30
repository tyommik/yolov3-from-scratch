import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, input_height, input_width, anchors, num_classes, CUDA = False):
    # TODO make it works with non-square aspect ratio
    batch_size = prediction.shape[0] # 1
    stride = input_height // prediction.shape[2] # 32
    grid_size = input_height // stride # 19
    bbox_attrs = 5 + num_classes # 85
    num_anchors = len(anchors) # 3

    # (1, 255, 13, 13) -> (1, 255, 169)
    prediction = prediction.reshape(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    # (1, 255, 169) -> (1, 169, 255)
    prediction = prediction.transpose(1, 2)
    # (1, 169, 255) -> (1, 507, 85)
    prediction = prediction.reshape(batch_size, -1, bbox_attrs)

    # every C has (5 + 80) length.
    # 5 = (tx, ty, tw, th, p0)
    # p0 is objectness score. The propability that an object is contained inside a bbox
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    anchors = [(one / stride, two / stride) for (one, two) in anchors]

    # Make grid
    # 1 bbox - (0, 0)
    # 2 bbox - (0, 0)
    # 3 bbox - (0, 0)
    # 1 bbox - (1, 0)
    # 2 bbox - (1, 0)
    # 3 bbox - (1, 0)
    # ...
    offset_x_y = np.array([[i] * 3 for i in range(grid_size)]).reshape(1, -1)
    offset_x_y_right = offset_x_y.repeat(grid_size).reshape(-1, 1)
    offset_x_y_right = torch.FloatTensor(offset_x_y_right)

    offset_x_y_left = np.array([offset_x_y] * grid_size)
    offset_x_y_left = offset_x_y_left.squeeze(1).reshape(-1, 1)
    offset_x_y_left = torch.FloatTensor(offset_x_y_left)

    if CUDA:
        offset_x_y_right = offset_x_y_right.cuda()
        offset_x_y_left = offset_x_y_left.cuda()

    offset_x_y = torch.cat((offset_x_y_left, offset_x_y_right), 1).unsqueeze(0)

    # sigma(tx) + Cx
    # sigma(ty) + Cy
    prediction[:,:,:2] += offset_x_y

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # (3, 2) -> (3 * 19 * 19, 2) -> (1083, 2) -> (1, 1083, 2)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)

    # exp(tw) * anchors
    # exp(th) * anchors
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # pred of class
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # real size
    prediction[:, :, :4] *= stride

    return prediction



