from typing import List, Tuple, Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from utils import predict_transform


def parse_cfg(cfg_file: str) -> List[Dict]:
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    with open(cfg_file, 'r') as cfg:
        blocks = []
        block = {}
        for line in cfg:
            line = line.strip()
            if line.startswith(('#', '\n')) or len(line) <= 0:
                continue
            if line.startswith("["):
                if block:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.rsplit('=')
                value = value.strip()
                key = key.strip().replace(' ', '')
                if value.isnumeric():
                    value = int(value)
                elif not value.isalpha():
                    value = list(map(float, value.split(',')))
                    if len(value) == 1:
                        value = value[0]

                block[key] = value
        blocks.append(block)
        return blocks


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def conv_bn(index, in_channels, out_channels, conv, activation, use_batchnorm=True, *args, **kwargs) -> nn.Sequential:
    """
    Make a layer

    :param index: str,  for layer name
    :param in_channels: int, input channels
    :param out_channels: int, output channels
    :param conv: nn.Module, Conv layer instance
    :param activation: nn.Module, activation function
    :param use_batchnorm: bool, use or  not batchnorm
    :param args:
    :param kwargs:
    :return: nn.Sequential
    """
    if use_batchnorm:
        return nn.Sequential(OrderedDict({
            f'conv_{index}': conv,
            f'bn_{index}': nn.BatchNorm2d(out_channels),
            f'activation_{index}': activation()
        }))
    else:
        return nn.Sequential(OrderedDict({
            f'conv_{index}': conv,
            f'activation_{index}': activation()
        }))


def create_modules(blocks: List[Dict]) -> Tuple[Dict, nn.Module]:
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filter = 3 # 3 - number of channels for images (RGB)
    output_filters = []

    for layer_idx, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
        if layer['type'] == 'convolutional':
            activation = layer['activation']

            batch_normalize = False
            bias = True
            if layer.get('batch_normalize'):
                batch_normalize = True
                bias = False

            filters = layer['filters']
            kernel_size = layer['size']
            padding = layer['pad']
            stride = layer['stride']

            conv = Conv2dAuto(in_channels=prev_filter,
                              out_channels=filters,
                              kernel_size=kernel_size,
                              stride=stride)
            module = conv_bn(layer_idx, prev_filter, filters, conv, activation=nn.LeakyReLU, use_batchnorm=batch_normalize)

        elif layer['type'] == 'maxpool':
            raise NotImplementedError

        elif layer['type'] == 'upsample':
            stride = layer['stride']
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module = nn.Sequential(OrderedDict({
                f'updample_{layer_idx}': upsample
            }))
        elif layer['type'] == 'route':
            if isinstance(layer['layers'], list):
                start = int(layer['layers'][0])
                end = int(layer['layers'][1])
            else:
                start = int(layer['layers'])
                end = 0

            #Positive anotation
            if start > 0:
                start = start - layer_idx
            if end > 0:
                end = end - layer_idx
            route = EmptyLayer()
            module.add_module("route_{0}".format(layer_idx), route)
            if end < 0:
                filters = output_filters[layer_idx + start] + output_filters[layer_idx + end]
            else:
                filters = output_filters[layer_idx + start]

        elif layer['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{layer_idx}', shortcut)

        elif layer['type'] == 'yolo':
            mask = [int(x) for x in layer['mask']]
            anchors = [int(x) for x in layer['anchors']]
            anchors = list(zip(anchors[::2], anchors[1::2]))
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors=anchors)
            module.add_module(f'YOLO_{layer_idx}', detection)
        else:
            raise NotImplementedError

        module_list.append(module)
        prev_filter = filters
        output_filters.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = parse_cfg(cfg)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {} # We cache the outputs for the route layer
        write = False
        for module_idx, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[module_idx](x)
            elif module_type == 'route':
                layers = module['layers']
                if isinstance(layers, list):
                    if layers[0] < 0:
                        layers[0] += module_idx
                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]

                    x = torch.cat((map1, map2), 1)

                else:
                    x = outputs[module_idx + layers]

            elif module_type == 'shortcut':
                map_from = outputs[module_idx + module['from']]
                map_prev = outputs[module_idx - 1]
                x = map_prev + map_from

            elif module_type == 'yolo':
                anchors = self.module_list[module_idx][0].anchors
                # get dim
                input_height = self.net_info['height']
                input_width = self.net_info['width']

                # get number of classes
                num_classes = module['classes']

                # transform
                x = x.data
                x = predict_transform(x, input_height, input_width, anchors, num_classes, CUDA=False)

                if not write:
                    detections = x
                    write = True

                else:
                    # so that's a detection layer
                    detections = torch.cat((detections, x), 1)

            outputs[module_idx] = x

        return x


if __name__ == '__main__':
    blocks = parse_cfg('../cfg/yolov3.cfg')
    a = create_modules(blocks)
    model = Darknet(cfg='../cfg/yolov3.cfg')
    print(model(torch.ones(1, 3, 608, 608)))