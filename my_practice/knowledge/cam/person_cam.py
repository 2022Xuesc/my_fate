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


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradients(self, grad):
        self.gradients.append(grad)

    # 给定输入x，调用
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                # 对层输出注册反向传播时的钩子函数
                x.register_hook(self.save_gradients)
                outputs += [x]
        return outputs, x  # 这里outputs是最后一个卷积层经过ReLU后的激活值


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
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        # self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)

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
        # self.model.features._modules['34'].weight.grad
        # Todo: 过程中收集到的是feature map激活值的梯度，而不是卷积参数的梯度！
        one_hot.backward()
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        # 共4个维度(batch_size,channel_size,h,w)，整合第2维和第3维度求平均，保留前两个维度；然后再取出第1维，得到每个通道的权重
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        maxL2Norm = 0
        maxIndex = 0
        for i, w in enumerate(weights):
            ch_cam = np.linalg.norm(w * target[i, :, :], ord=2)
            if ch_cam > maxL2Norm:
                maxL2Norm = ch_cam
                maxIndex = i
            # cam += w * target[i,:,:]
        # cam = np.maximum(cam, 0)  # 进行ReLU
        # cam = cv2.resize(cam, (224, 224))  # 上采样到(224,224)
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)  # 再进行归一化
        # print('最显著的通道是', maxIndex)
        return maxIndex


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

if __name__ == '__main__':
    # 最外层遍历target_layers
    target_layers = ["2","4","5","6","7"]
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
    # 写到文件中
    for target_layer in target_layers:
        # Todo: 使用训练好的全局模型
        grad_cam = GradCam(model=model, target_layer_names=[target_layer], use_cuda=True)
        times = [0] * 2048
        files = os.listdir(dir_name)
        n = len(files)
        cnt = 100
        for i in range(n):
            file_name = files[i]
            if not file_name.startswith("COCO"):
                continue
            image_path = os.path.join(dir_name, file_name)
            # Todo: 原来的图像读取方法
            # 新的图像读取方法
            input = Image.open(image_path).convert('RGB')
            input = transforms(input)
            input.unsqueeze_(0)
            target_index = 49
            index = grad_cam(input, target_index)
            times[index] += 1
            # print(f'progress: {i} / {n}')
            if i == cnt - 1:
                break

        times = [item / cnt for item in times]
        top_k_values,top_k_indices = top_k_values_with_indices(times,k=10)
        print(top_k_values)
        print(top_k_indices)
        print("*" * 50)
    # 使用cv2处理
    # img = cv2.imread(image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # show_cam_on_image(img, mask)

# if __name__ == '__main__':
#     # Todo: 找图像
#     valid_path = "my_imgs"
#     category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'
#     device = 'cpu'
#     model = create_resnet101_model(pretrained=False,device=device)
#     # 读取np数组数据
#     arrs = np.load('global_model.npy',allow_pickle=True)
#     agg_tensors = []
#     for arr in arrs:
#         agg_tensors.append(torch.from_numpy(arr).to(device))
#     for param,agg_tensor in zip(model.named_parameters(),agg_tensors):
#         param[1].data.copy_(agg_tensor)
#     # Todo: 验证全局模型的性能
#     valid_dataset = COCO(
#         valid_path,
#         config_dir=category_dir,
#         transforms=valid_transforms()
#     )
#     batch_size=1
#     valid_loader = DataLoader(
#         dataset=valid_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4
#     )
#     model.eval()
#     sigmoid_func = torch.nn.Sigmoid()
#     for valid_step,(inputs,target) in enumerate(valid_loader):
#         inputs = inputs.to(device)
#         target = target.to(device)
#         output = model(inputs)
#         sig_output = sigmoid_func(output)
