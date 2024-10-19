# 服务器与客户端的通用逻辑
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

import json
import csv
import random

import os
import os.path
import pickle


# import util
# from util import *


def findMaxIndex(arr, S, cur):
    maxVal = 0
    maxIndex = -1
    for i in range(len(arr)):
        if i not in S and i != cur:
            if arr[i] > maxVal:
                maxVal = arr[i]
                maxIndex = i
    return maxIndex

    # features是该批次中的特征
    # predicts是该批次的预测值


# 给定该批次的样本的预测向量，从每个预测向量根据标签相关性重构出其他的标签向量
def getCandidates(predicts, adjList, requires_grad):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    candidates = torch.zeros((batch_size, label_dim), dtype=torch.float64).to(device)
    # Todo: 将以下部分封装成一个函数，从其他标签的向量出发得到
    for b in range(batch_size):
        predict_vec = predicts[b]  # 1 * C维度
        # 遍历每一个推断出来的标签
        for lj in range(label_dim):
            relation_num = 0
            for li in range(label_dim):
                # 判断从li是否能推断出lj
                if lj in adjList[li]:
                    # a表示从li到lj的相关性，不是计算损失，无需使用带有梯度的相关性值
                    if requires_grad:
                        a = adjList[li][lj]
                    else:
                        a = adjList[li][lj].item()
                    # 需要进行归一化，归一化系数为len(adjList[li])
                    candidates[b][lj] += predict_vec[li] * a
                    relation_num += 1
                elif li == lj:
                    candidates[b][lj] += predict_vec[li]
                    relation_num += 1
            candidates[b][lj] /= relation_num
    return candidates


def LabelOMP(predicts, adjList):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    # 每张图片，找和它最相似的k张图片
    # Todo: 对这里相似图片的张数进行修改
    k = batch_size // 2

    # 最终输出的是图像之间的语义相似度矩阵
    predict_similarities = torch.zeros(batch_size, batch_size, dtype=torch.float64).to(device)

    # 遍历每张图片
    for i in range(batch_size):
        # 需要拟合的残差
        predict = predicts[i]

        # 进行k次迭代
        # Todo: 维护相似集合，以及相似图像的特征向量和预测向量
        S = set()
        indexes = []
        candidateX = torch.empty(0, label_dim, dtype=torch.float64).to(device)

        # candidates表示从一个标签预测向量中根据标签相关性推断出来的新预测向量
        candidates = getCandidates(predicts, adjList, requires_grad=False)
        # 现在可以计算内积了
        candidate_inner_products = torch.matmul(candidates, predict)
        # 对第1维计算范数
        candidate_norms = torch.norm(candidates, dim=1)
        predict_scores = candidate_inner_products / candidate_norms
        for j in range(k):
            # 找到最相似的图像i‘
            # 从中选出相似性最高的图像，加到相似集中
            index = findMaxIndex(predict_scores, S, i)
            # 判断是否满足内积大于等于0的条件
            # 如果不满足，说明找不到相似图片，直接退出即可
            if torch.matmul(predicts[index], predict) <= 0:
                break
            S.add(index)
            indexes.append(index)

            candidateX = torch.cat((candidateX, candidates[index].unsqueeze(0)), dim=0)

            # Todo: 这里不应该预测值拟合，而应该是预测值经相关性的拟合？
            predict_coefficients = torch.linalg.lstsq(torch.transpose(candidateX, 0, 1), predict)[0]
            # 更新相似性矩阵
            # 不是对称的，因此，更新第i行
            for m in range(len(indexes)):
                neighbor = indexes[m]
                # Todo: 验证引入的约束操作对与原解的修改情况
                predict_similarities[i][neighbor] = max(0, predict_coefficients[m])  # 确保相似性大于0
    return predict_similarities.detach()  # 对于无需通过pytorch计算图优化的变量，将其detach


class LabelSmoothLoss(nn.Module):
    # 进行相关参数的设置
    def __init__(self, relation_need_grad=False):
        super(LabelSmoothLoss, self).__init__()
        self.relation_need_grad = relation_need_grad

    # 传入
    # 1. 预测概率y：b * 80
    # 2. 图像语义相似度: b * b
    # 3. 标签相关性: 80 * 80
    def forward(self, predicts, similarities, adjList):
        batch_size = len(predicts)
        # 相似图像的个数是 batch_size // 2
        # Todo: 对于每个样本，如果没有相似的图像，则不考虑这个图像的标签平滑损失
        total_loss = 0
        cnt = 0
        candidates = getCandidates(predicts, adjList, self.relation_need_grad)
        for i in range(batch_size):
            if torch.sum(similarities[i]) == 0:
                continue
            cnt += 1
            total_loss += torch.norm(predicts[i] - torch.matmul(similarities[i], candidates), p=2)
        return 0 if cnt == 0 else total_loss / cnt


class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # 计算预测为正标签的概率和预测为负标签的概率
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            # 这里使用原地操作
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # 基本的交叉熵损失计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        # 和未优化的版本逻辑相同
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class AveragePrecisionMeter(object):
    """
    计算每个类（标签）的平均精度
    给定输入为:
    1. N*K的输出张量output：值越大，置信度越高;
    2. N*K的目标张量target：二值向量，0表示负样本，1表示正样本
    3. 可选的N*1权重向量：每个样本的权重
    N是样本个数，K是类别即标签个数
    """

    # Todo: 这里difficult_examples的含义是什么？
    #  可能存在难以识别的目标（模糊、被遮挡、部分消失），往往需要更加复杂的特征进行识别
    #  为了更加有效评估目标检测算法的性能，一般会对这些目标单独处理
    #  标记为difficult的目标物体可能不会作为正样本、也不会作为负样本，而是作为“无效”样本，不会对评价指标产生影响
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """将计量器的成员变量重置为空"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor，每个样本对应的每个标签的预测概率向量，和为1
            target (Tensor): binary NxK tensor，表示每个样本的真实标签分布
            weight (optional, Tensor): Nx1 tensor，表示每个样本的权重
        """

        # Todo: 进行一些必要的维度转换与检查
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # 确保存储有足够的大小-->对存储进行扩容
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # 存储预测分数scores和目标值targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """ 返回每个类的平均精度
        Return:
            ap (FloatTensor): 1xK tensor，对应标签（类别）k的平均精度
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        # non_zero_labels = 0
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            # if targets.sum() != 0:
            #     non_zero_labels += 1
        # Todo: 在这里判断不为空的标签个数，直接求均值
        # return ap.sum() / non_zero_labels
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=False):

        # 对输出概率进行排序
        # Todo: 这里第0维是K吗？跑一遍GCN进行验证
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # 计算prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        # 遍历排序后的下标即可
        for i in indices:
            label = target[i]
            # Todo: 如果是设置跳过较难预测的样本并且标签为0，则跳过
            if difficult_examples and label == 0:
                continue
            # 更新正标签的个数
            if label == 1:
                pos_count += 1
            # 更新已遍历总标签的个数
            total_count += 1
            if label == 1:
                # 说明召回水平增加，计算precision
                precision_at_i += pos_count / total_count
        # 除以样本的正标签个数对精度进行平均
        # Todo: 一般不需要该判断语句，每个样本总有正标签
        # if pos_count != 0:
        precision_at_i /= pos_count
        # 返回该样本的average precision
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


# if set == "test":
#     self.path_images = '/home/klaus125/research/dataset/VOC2007_Expanded/VOCdevkit/VOC2007/JPEGImages'
# else:
#     self.path_images = '/home/klaus125/research/dataset/VOC2007/JPEGImages/origin'
class Voc2007Classification(Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, ):
        self.root = root
        self.set = set
        self.path_images = os.path.join(root, set)
        self.transform = transform
        self.target_transform = target_transform

        # define path of csv file
        path_csv = os.path.join(self.root, 'csvs')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


def create_resnet101_model(pretrained, device, num_classes=80):
    model = torch_models.resnet101(pretrained=True, num_classes=1000)
    # 将最后的全连接层替换掉
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_classes))
    torch.nn.init.kaiming_normal_(model.fc[0].weight.data)
    return model.to(device)


def construct_relation_by_matrix(matrix, device):
    adjList = [dict() for _ in range(num_labels)]
    variables = []
    # 保留连接强度的阈值
    th = 0.2
    for i in range(num_labels):
        for j in range(num_labels):
            # 自相关性如何处理？额外进行处理
            if i == j or matrix[i][j] < th:
                continue
            variable = torch.tensor(matrix[i][j]).to(device)
            variable.requires_grad_()
            variables.append(variable)
            # 直接维护对应的优化变量即可
            adjList[i][j] = variable
    relation_optimizer = torch.optim.SGD(variables, lr=0.1)
    # 返回优化变量组、邻接表以及标签相关性的优化器
    return variables, adjList, relation_optimizer


class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


train_transforms = transforms.Compose([
    # 这里是(512,512)，而不是(512)
    transforms.Resize((512, 512)),
    MultiScaleCrop(448, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 相关训练参数的设置，和client1的配置相一致
batch_size = 8
device = 'cuda:0'
learning_rate = 0.0001

# 既定的配置，一般不会发生变化
num_workers = 32
drop_last = False
shuffle = True
num_labels = 20
momentum = 0.9

root_path = "/home/klaus125/research/dataset/voc_standalone"
train_dataset = Voc2007Classification(root_path, "trainval")
valid_dataset = Voc2007Classification(root_path, "test")

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
    drop_last=drop_last, shuffle=shuffle
)

model = create_resnet101_model(pretrained=True, device=device, num_classes=num_labels)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
# 交叉熵损失，具体来说，是非对称损失
# criterion = AsymmetricLossOptimized().to(device)
criterion = nn.MultiLabelSoftMarginLoss()
# 标签损失的正则因子
# lambda_y = 1

# label_smooth_loss = LabelSmoothLoss()
ap_meter = AveragePrecisionMeter(difficult_examples=True)

# Todo: 更改，40或者20
epochs = 200

# one_cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#                                                    max_lr=learning_rate,
#                                                    epochs=epochs,
#                                                    steps_per_epoch=len(train_loader),
#                                                    verbose=False)

# 使用reduceLROnPlateau学习率调度器
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2)

# 构建优化参数
image_id2labels = json.load(open(json_file, 'r'))
adjMatrix = np.zeros((num_labels, num_labels))
nums = np.zeros(num_labels)
for image_info in image_id2labels:
    labels = image_info['labels']
    for label in labels:
        nums[label] += 1
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            x = labels[i]
            y = labels[j]
            adjMatrix[x][y] += 1
            adjMatrix[y][x] += 1
nums = nums[:, np.newaxis]
# 遍历每一行
for i in range(num_labels):
    if nums[i] != 0:
        adjMatrix[i] = adjMatrix[i] / nums[i]
# 遍历A，将主对角线元素设置为1
for i in range(num_labels):
    adjMatrix[i][i] = 1

# 从矩阵中构建出优化参数组
variables, adjList, relation_optimizer = construct_relation_by_matrix(matrix=adjMatrix, device=device)

# base = 0.0001
# for i in range(100):
#     print(base)
#     base *= 0.9

# 开始执行优化过程
for epoch in range(0, epochs):
    total_samples = len(train_loader.sampler)
    steps_per_epoch = math.ceil(total_samples / batch_size)
    ap_meter.reset()
    model.train()
    epoch_loss = 0
    # 直接输出信息到控制台上
    for train_step, (inputs, target) in enumerate(train_loader):
        print(f'正在训练第 {train_step} / {steps_per_epoch} 个 batch')
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)

        ap_meter.add(output.data, target.data)

        predicts = torch.sigmoid(output).to(torch.float64)

        predict_similarities = LabelOMP(predicts.detach(), adjList)

        label_loss = LabelSmoothLoss(relation_need_grad=True)(predicts.detach(), predict_similarities,
                                                              adjList)

        if label_loss != 0:
            relation_optimizer.zero_grad()
            label_loss.backward()
            relation_optimizer.step()
        for variable in variables:
            variable.data = torch.clamp(variable.data, min=0.0, max=1.0)

        loss = criterion(output, target) + \
               lambda_y * LabelSmoothLoss(relation_need_grad=False)(predicts, predict_similarities,
                                                                    adjList)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        print("==============================================")
        pre_op_lr = optimizer.param_groups[0]['lr']
        # print(f'AdamW 调整前学习率为: {pre_op_lr}')
        optimizer.step()
        after_op_lr = optimizer.param_groups[0]['lr']
        # print(f"AdamW 调整后学习率为: {after_op_lr}")
        # Todo: 输出一下学习率的变化
        # one_cycle_scheduler.step()
    epoch_loss /= steps_per_epoch
    print(f"epoch = f{epoch}时候，损失为：{epoch_loss}")
    plateau_scheduler.step(epoch_loss)
    after_scheduler_lr = optimizer.param_groups[0]['lr']
    print(f'当前epoch学习率为： {after_scheduler_lr}')
