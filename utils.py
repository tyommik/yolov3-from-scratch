import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.shape[0] # 1
    stride = inp_dim // prediction.shape[2] # 46
    grid_size = inp_dim // stride # 13
    bbox_attrs = 5 + num_classes # 85
    num_anchors = len(anchors) # 3

    # (1, 255, 13, 13) -> (1, 255, 169)
    prediction = prediction.reshape(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    # (1, 255, 169) -> (1, 169, 255)
    prediction = prediction.transpose(1, 2)
    # (1, 169, 255) -> (1, 507, 85)
    prediction = prediction.reshape(batch_size, -1, bbox_attrs)


