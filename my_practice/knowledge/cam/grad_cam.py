import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse


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
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # 获取给定输入x时网络的目标层的激活值，以及网络的输出
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        # 经过分类器层
        output = self.model.classifier(output)
        return target_activations, output


# 对图像进行预处理
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
    preprocessed_img.unsqueeze_(0)
    # 封装到Variable中
    input = Variable(preprocessed_img, requires_grad=True)
    return input


# 输入图像和mask(归一化的grad cam图)
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('dog_grad_cam.jpg', np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
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
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # output.size()[-1]是类别数
        one_hot[0][index] = 1  # 若index为None，则one_hot会变成一个全1向量
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # self.model.features._modules['34'].weight.grad
        # Todo: 过程中收集到的是feature map激活值的梯度，而不是卷积参数的梯度！
        one_hot.backward()
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        # 共4个维度(batch_size,channel_size,h,w)，整合第2维和第3维度求平均，保留前两个维度；然后再取出第1维，得到每个通道的权重
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)  # 进行ReLU
        cam = cv2.resize(cam, (224, 224))  # 上采样到(224,224)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)  # 再进行归一化
        return cam


class GuidedBackpropReLU(Function):
    def forward(self, intput):
        # 创建一个和input形状相同的张量positive_mask，其中大于0的元素为1，小于等于0的元素为0
        # 捕获ReLU中的正激活部分
        positive_mask = (input > 0).type_as(input)
        # torch.addcmul(t,t1,t2)，将t1和t2逐元素相乘，加到t上
        # 本代码中，得到ReLU函数的输出
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        # 将输入张量input和ReLU函数的输出output保存在saved_tensors中，以便反向传播时使用
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        # 反向传播时，读取之前保存的输入和输出向量
        input, output = self.saved_tensors
        # 输入的掩码矩阵
        positive_mask_1 = (input > 0).type_as(grad_output)
        # 输出梯度的掩码矩阵
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        # 计算输入张量对于梯度张量的反向传播梯度
        # Todo: 相当于输入梯度 = 输出梯度 * 输出梯度的掩码矩阵 * 输入掩码矩阵
        #  反向传播过程中，传播ReLU函数的正激活部分对应的梯度，而将负激活部分的梯度置为0
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)
        # 最后返输入的梯度
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        # 使用GuidedBackpropReLU替换ReLU
        # Todo: 原先ReLU的梯度传播时的行为是什么呢？
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)
        # 获取输出最大的类别索引
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        one_hot.backward()
        # 获取输入的cam？
        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        return output


if __name__ == '__main__':
    image_path = "dog.jpeg"
    # 假设模型有特征层和分类器层
    # 下面以VGG模型为例
    grad_cam = GradCam(model=models.vgg19(pretrained=True), target_layer_names=["35"], use_cuda=True)
    img = cv2.imread(image_path, 1)
    # 先resize到期望的输入规模，并进行归一化
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    # 如果为None，返回最高得分类别的map，否则，指定类别索引
    target_index = None
    mask = grad_cam(input, target_index)
    show_cam_on_image(img, mask)

    # Todo: 调试好带有定制反向传播的CAM
    gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=True)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')