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

hook_features = []
hook_gradients = []


def save_gradient(grad):
    hook_gradients.append(grad)


def save_outputs(module, input, output):
    hook_features.append(output.data)
    # Todo: 这里应该可以注册梯度函数
    output.register_hook(save_gradient)


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradients(self, grad):
        self.gradients.append(grad)

    # 给定输入x，调用
    def __call__(self, x):
        self.gradients = []
        cnt = 0
        # 遍历所有ReLU层，注册钩子函数
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(save_outputs)
            if isinstance(module, nn.Conv2d):
                cnt += 1
        # Todo: relu层的个数和conv层的个数并不一致
        for name, module in self.model._modules.items():
            x = module(x)
        # # 为所有ReLU层注册钩子函数
        # for module in self.model.modules():
        #     x =
        return hook_features, x  # 这里outputs是最后一个卷积层经过ReLU后的激活值


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        # 定义特征提取器
        # 提取出resnet101的特征层
        modules = list(model.children())
        self.features = nn.Sequential(*modules[:-1])
        self.feature_extractor = FeatureExtractor(self.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # 获取给定输入x时网络的目标层的激活值，以及网络的输出
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        # 经过分类器层
        output = self.model.fc(output)
        return target_activations, output


def preprocess_image(img):
    # 对通道进行处理
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # 对图像的颜色通道进行反转
    # opencv等库一般使用BGR通道顺序，而预处理使用RGB通道顺序
    preprocessed_img = img.copy()[:, :, ::-1]
    # 对每个通道，减去均值，并除以标准差
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    # 转置图像，将颜色通道维度放在最前面
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    # 添加批次batch_size维度
    # preprocessed_img.unsqueeze_(0)
    # 封装到Variable中
    # input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('person_cam.jpg', np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, ratio, use_cuda, target_layer_names=None):
        self.model = model
        # self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)
        self.ratio = ratio

    def forward(self, intput):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        # Todo: 这里需要适配多标签？
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # output.size()[-1]是类别数
        one_hot[0][index] = 1  # 若index为None，则one_hot会变成一个全1向量
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.extractor.features.zero_grad()
        self.model.fc.zero_grad()
        # Todo: 过程中收集到的是feature map激活值的梯度，而不是卷积参数的梯度！
        one_hot.backward()

        # 遍历每一个需要统计的层
        indices = []

        for i in range(len(features)):
            # 获取对应层的梯度值
            grads_val = hook_gradients[-i - 1].cpu().data.numpy()
            target = features[i]
            target = target.cpu().data.numpy()[0, :]
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            # Todo: 对于较深的层，不为0的通道数很少
            ch_cams = []
            # 使用的relu层后的，可以保证激活映射是非负的，但是梯度可能是负数
            # 那这样不能直接使用L2范数，而应该进行relu后再使用L2范数？
            for j, w in enumerate(weights):
                ch_cam = w * np.linalg.norm(target[j, :, :], ord=2)
                ch_cams.append(ch_cam)
            k = int(self.ratio * len(ch_cams))
            top_values, top_indices = top_k_values_with_indices(ch_cams, k=k)
            indices.append(top_indices)
        return indices

        # grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #
        # target = features[-1]
        # target = target.cpu().data.numpy()[0, :]
        # # 共4个维度(batch_size,channel_size,h,w)，整合第2维和第3维度求平均，保留前两个维度；然后再取出第1维，得到每个通道的权重
        # weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # # cam = np.zeros(target.shape[1:], dtype=np.float32)
        # # Todo: 通道全是0
        # ch_cams = []
        # for i, w in enumerate(weights):
        #     ch_cam = np.linalg.norm(w * target[i, :, :], ord=2)
        #     ch_cams.append(ch_cam)
        #     # cam += w * target[i,:,:]
        # # cam = np.maximum(cam, 0)  # 进行ReLU
        # # cam = cv2.resize(cam, (224, 224))  # 上采样到(224,224)
        # # cam = cam - np.min(cam)
        # # cam = cam / np.max(cam)  # 再进行归一化
        # # print('最显著的通道是', maxIndex)
        # # Todo: 剪枝掉前10%的通道
        # k = int(self.ratio * len(ch_cams))
        # # k = 10
        # top_values, top_indices = top_k_values_with_indices(ch_cams, k=k)
        # return top_indices


# 数据集处理方式
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


def valid_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        # 输入图像是224*224
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


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


def write_lists_to_file(lists, filename):
    with open(filename, 'a') as file:
        for lst in lists:
            file.write(' '.join(map(str, lst)) + '\n')


def init_list(lst):
    newList = []
    for i in range(len(lst)):
        newList.append([0] * len(lst[i]) * 2)
    return newList


if __name__ == '__main__':
    # 最外层遍历target_layers
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
    # num_channels = [64, 256, 512, 1024, 2048]
    all_layers = []
    dir_name = 'my_imgs'
    device = 'cuda:0'
    model = create_resnet101_model(pretrained=False, device=device)

    model.eval()
    # 读取np数组数据
    arrs = np.load('global_model.npy', allow_pickle=True)
    agg_tensors = []
    for arr in arrs:
        agg_tensors.append(torch.from_numpy(arr).to(device))
    for param, agg_tensor in zip(model.named_parameters(), agg_tensors):
        param[1].data.copy_(agg_tensor)
    transforms = valid_transforms()

    # 每次计算所有的层
    ratio = 0.5
    grad_cam = GradCam(model=model, ratio=ratio, use_cuda=True, target_layer_names=target_layers)
    files = os.listdir(dir_name)
    n = len(files)
    cnt = 1
    # 遍历每张图片，计算显著的通道
    # times = [[0] * num_channel for num_channel in num_channels]
    times = []
    candidates = dict()
    for name, module in model.named_modules():
        if "conv" in name:
            all_layers.append(name)
    for i in range(n):
        # 统计的全局信息清空
        hook_features.clear()
        hook_gradients.clear()

        file_name = files[i]
        if not file_name.startswith("COCO"):
            continue
        image_path = os.path.join(dir_name, file_name)
        # 新的图像读取方法
        input = Image.open(image_path).convert('RGB')
        input = transforms(input)
        input.unsqueeze_(0)
        target_index = 49
        # 这个应该是多层多通道的
        indexes = grad_cam(input, target_index)
        # 统计该图像对times数组的贡献
        # 如果times为空,则根据indexes的形状声称
        if len(times) == 0:
            times = init_list(indexes)
        # 遍历每一层
        for j in range(len(times)):
            for k in range(len(indexes[j])):
                times[j][indexes[j][k]] += 1
        if i == cnt - 1:
            break
    # 所有times数组统计结束后，使用cnt进行归一化
    for i in range(len(times)):
        layer_name = all_layers[i]
        # 如果layer_name不在目标层中，则直接跳过
        if layer_name not in target_layers:
            continue
        times[i] = [item / cnt for item in times[i]]
        pruned_nums = int(ratio * len(times[i]))
        top_k_values, top_k_indices = top_k_values_with_indices(times[i], k=pruned_nums)
        end = len(top_k_values)
        if top_k_values.__contains__(0):
            end = top_k_values.index(0)
        candidate = top_k_indices[:min(end, pruned_nums)]
        candidates[layer_name] = candidate
    # 将字典写入文件中
    file_path = 'data.json'
    with open(file_path, 'w') as json_file:
        json.dump(candidates, json_file)
