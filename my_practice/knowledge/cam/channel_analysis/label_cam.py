import copy
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from federatedml.nn.backend.multi_label.models import *
import pickle
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import math
import torch
import torch.nn as nn
import torchvision.models as torch_models
from PIL import Image
from torch.utils.data import Dataset

from federatedml.nn.backend.multi_label.meta_learning.maml import MAML

import json
import os
import pickle


# 一些utils方法
def top_k_values_with_indices(lst, k):
    # 使用 sorted() 对列表进行排序，并指定 reverse=True 获取降序排序
    sorted_lst = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

    top_k_values = []
    top_k_indices = []

    for i in range(k):
        index, value = sorted_lst[i]
        top_k_values.append(value)
        top_k_indices.append(index)

    return top_k_values, top_k_indices


def init_list(lst):
    newList = []
    for i in range(len(lst)):
        newList.append([0] * len(lst[i]) * 2)
    return newList


def valid_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        # 输入图像是224*224
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class COCO(Dataset):
    def __init__(self, images_dir, config_dir, transforms=None, inp_name=None):
        self.images_dir = images_dir
        self.config_dir = config_dir
        self.transforms = transforms
        self.img_list = []
        self.cat2idx = None
        self.get_anno()

        self.num_classes = len(self.cat2idx)
        self.inp = None
        if inp_name is not None:
            inp_file = os.path.join(self.config_dir, inp_name)
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.images_dir, 'anno.json')
        self.img_list = json.load(open(list_path, 'r'))
        category_path = os.path.join(self.config_dir, 'category.json')
        self.cat2idx = json.load(open(category_path, 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img, target = self.get(item)
        # 如果self.inp不为空，说明是在GCN的配置环境下
        if self.inp is not None:
            return (img, self.inp), target
        else:  # 否则使用的是常规的网络，直接返回img和target即可
            return img, target

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        # 读取图像数据
        img = Image.open(os.path.join(self.images_dir, filename)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # Todo: 这里负标签设置为0，正标签设置为1
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return img, target


hook_features = []
hook_gradients = []
hooks = []


# Todo: 验证收集到的feature map和gradient是否一致
# GradCam需要保存feature map的梯度
def save_gradient(grad):
    hook_gradients.append(grad)


# 保存feature map的值
def save_outputs(module, input, output):
    hook_features.append(output.data)
    output.register_hook(save_gradient)


class FeatureExtractor():
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        # 遍历所有ReLU层，注册钩子函数
        cnt = 0
        # 如果还没有注册钩子函数，则进行注册
        if len(hooks) == 0:
            for module in self.model.modules():
                # Todo: 关于ReLU和Conv2d的对应关系
                if isinstance(module, nn.ReLU):
                    hooks.append(module.register_forward_hook(save_outputs))
                # if isinstance(module, nn.Conv2d):  # Todo: downsample后没有经过ReLU，但也属于conv2d，这里暂时不对其数据进行统计
                #     cnt += 1
        for name, module in self.model._modules.items():
            x = module(x)
        # 取消注册
        return hook_features, x


class ModelOutputs():
    def __init__(self, model):
        self.model = model
        modules = list(model.children())
        # 将特征层封装到一起
        self.features = nn.Sequential(*modules[:-1])
        self.feature_extractor = FeatureExtractor(self.features)

    def __call__(self, x):
        # 获取目标层的激活值，以及网络的输出
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.fc(output)
        return target_activations, output


class GradCam:
    # ratio表示选择显著通道的比例
    def __init__(self, model, ratio, use_cuda):
        self.model = model
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model)
        self.ratio = ratio

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # output.size()[-1]是类别数
        one_hot[0][index] = 1  # 若index为None，则one_hot会变成一个全1向量
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.extractor.features.zero_grad()
        self.model.fc.zero_grad()
        # 过程中收集到的是feature map激活值的梯度，而不是卷积参数的梯度！
        one_hot.backward()

        # 这里返回在每个通道的权重
        # indices = []
        layer_cams = []
        # 遍历每一个需要统计的层
        for i in range(len(features)):
            # Todo: 这里逆序的原因是：features从浅到深添加，而gradients从深到浅添加
            grads_val = hook_gradients[-i - 1].cpu().data.numpy()
            target = features[i]
            target = target.cpu().data.numpy()[0, :]
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            # Todo: 对于较深的层，不为0的通道数很少
            ch_cams = np.zeros(len(weights))
            for j, w in enumerate(weights):
                ch_cams[j] = w * np.linalg.norm(target[j, :, :], ord=2)
            layer_cams.append(ch_cams)
        return layer_cams
            # k = int(self.ratio * len(ch_cams))
            # top_values, top_indices = top_k_values_with_indices(ch_cams, k=k)
            # indices.append(top_indices)
        # return indices


if __name__ == '__main__':
    # dir_name = '/home/klaus125/research/dataset/label_imgs'
    dir_name = '/data/projects/dataset/label_imgs'
    # Todo: 设置GPU卡
    device = 'cuda:5'
    model = create_resnet101_model(pretrained=False, device=device)
    # Todo: 注意该语句对模型输出的影响
    #  需要得到running_mean和running_var
    model.eval()

    all_layers = []

    # 读取训练好的全局模型
    # 读取bn层的统计数据
    arrs = np.load('../../../../state/global_model.npy', allow_pickle=True)
    agg_tensors = []
    for arr in arrs:
        agg_tensors.append(torch.from_numpy(arr).to(device))
    for param, agg_tensor in zip(model.named_parameters(), agg_tensors):
        param[1].data.copy_(agg_tensor)
    bn_data = np.load("../../../../state/bn_data.npy",allow_pickle=True)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.running_mean.data.copy_(torch.from_numpy(bn_data[idx]).to(device))
            idx += 1
            layer.running_var.data.copy_(torch.from_numpy(bn_data[idx]).to(device))
            idx += 1

    transforms = valid_transforms()

    # 目标层
    target_layers = ['layer4.2.conv3', 'layer3.22.conv3', 'layer2.3.conv3', 'layer1.2.conv3', 'conv1', 'layer1.0.conv2',
                     'layer1.0.conv1', 'layer1.1.conv2', 'layer1.1.conv1', 'layer1.2.conv2', 'layer1.2.conv1',
                     'layer2.0.conv2', 'layer2.0.conv1', 'layer2.1.conv2', 'layer2.1.conv1', 'layer2.2.conv2',
                     'layer2.2.conv1', 'layer2.3.conv2', 'layer2.3.conv1', 'layer3.0.conv2', 'layer3.0.conv1',
                     'layer3.1.conv2', 'layer3.1.conv1', 'layer3.2.conv2', 'layer3.2.conv1', 'layer3.3.conv2',
                     'layer3.3.conv1', 'layer3.4.conv2', 'layer3.4.conv1', 'layer3.5.conv2', 'layer3.5.conv1',
                     'layer3.6.conv2', 'layer3.6.conv1', 'layer3.7.conv2', 'layer3.7.conv1', 'layer3.8.conv2',
                     'layer3.8.conv1', 'layer3.9.conv2', 'layer3.9.conv1', 'layer3.10.conv2', 'layer3.10.conv1',
                     'layer3.11.conv2', 'layer3.11.conv1', 'layer3.12.conv2', 'layer3.12.conv1', 'layer3.13.conv2',
                     'layer3.13.conv1', 'layer3.14.conv2', 'layer3.14.conv1', 'layer3.15.conv2', 'layer3.15.conv1',
                     'layer3.16.conv2', 'layer3.16.conv1', 'layer3.17.conv2', 'layer3.17.conv1', 'layer3.18.conv2',
                     'layer3.18.conv1', 'layer3.19.conv2', 'layer3.19.conv1', 'layer3.20.conv2', 'layer3.20.conv1',
                     'layer3.21.conv2', 'layer3.21.conv1', 'layer3.22.conv2', 'layer3.22.conv1', 'layer4.0.conv2',
                     'layer4.0.conv1', 'layer4.1.conv2', 'layer4.1.conv1', 'layer4.2.conv2', 'layer4.2.conv1']

    ratio = 0.2
    grad_cam = GradCam(model=model, ratio=ratio, use_cuda=True)

    num_labels = 80
    statistics = []
    candidates = dict()

    for name, module in model.named_modules():
        if "conv" in name:
            all_layers.append(name)

    total = 0
    # 遍历每一个标签
    for label in range(num_labels):
        # 遍历模型，初始化
        layer_cams = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv' in name:
                # 记录输出通道数
                layer_cams.append(np.zeros(module.weight.size(0)))
        label_dir = os.path.join(dir_name, str(label))
        files = os.listdir(label_dir)
        n = len(files)
        cnt = 100
        total += cnt
        for i in range(n):
            # Todo: 打印进度
            print(f'label = {label} , progress = {i + 1} / {cnt}')
            hook_features.clear()
            hook_gradients.clear()
            file_name = files[i]
            if not file_name.startswith("COCO"):
                continue
            image_path = os.path.join(label_dir, file_name)
            # 新的图像读取方法
            input = Image.open(image_path).convert('RGB')
            input = transforms(input)
            input.unsqueeze_(0)
            target_index = label

            layer_cam = grad_cam(input, target_index)

            # 遍历每一层
            for j in range(len(layer_cams)):
                layer_cams[j] += layer_cam[j]

            # 每个标签只选取cnt张图片进行实验
            if i == cnt - 1:
                # Todo: 放到标签对应
                statistics.append(copy.copy(layer_cams))
                break
    with open('server_statistics.pkl', 'wb') as file:
        pickle.dump(statistics, file)
