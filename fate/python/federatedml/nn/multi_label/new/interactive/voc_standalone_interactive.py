# 服务器与客户端的通用逻辑
import math
import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torchnet.meter as tnt
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset

import csv
import json
import os
import os.path
import os.path
import pickle
import random
from collections import OrderedDict

method = 'single_interactive'

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


from scipy.optimize import shgo
def my_general_linear_model_func(A1, b1):
    num_x = np.shape(A1)[1]

    def my_func(x):
        ls = 0.5 * (b1 - np.dot(A1, x)) ** 2
        result = np.sum(ls)
        return result

    bnds = [(0, None)]
    for i in range(num_x - 1):
        bnds.append((0, None))
    res1 = shgo(my_func,
                bounds=bnds)

    return res1.x


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
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


# Todo: VOC还需要返回inp数据
class Voc2007Classification(Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, config_dir=None, inp_name=None):
        self.root = root
        self.set = set
        self.path_images = os.path.join(root, set)
        self.transform = transform
        self.target_transform = target_transform

        path_csv = os.path.join(self.root, 'csvs')
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        self.config_dir = config_dir
        self.inp = None
        if inp_name is not None:
            inp_file = os.path.join(self.config_dir, inp_name)
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.inp is not None:
            return (img, self.inp, index), target
        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


def findMaxIndex(arr, S, cur):
    maxVal = 0
    maxIndex = -1
    for i in range(len(arr)):
        if i not in S and i != cur:
            if arr[i] > maxVal:
                maxVal = arr[i]
                maxIndex = i
    return maxIndex


# 计算正确的候选者
def getCorrectedCandidates(predicts, adjList, label_prob_vec, requires_grad):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    candidates = torch.zeros((batch_size, label_dim), dtype=torch.float64).to(device)

    exists_lower_bound = 0.5 + predict_gap

    # 遍历每个批次
    for b in range(batch_size):
        # 输入1*C，输出1*C
        predict_vec = predicts[b]
        # 需要推断的标签
        for lj in range(label_dim):
            relation_num = 0

            incr_lower_bound = label_prob_vec[lj] + relation_gap
            decr_upper_bound = label_prob_vec[lj] - relation_gap
            for li in range(label_dim):
                if li == lj:  # 是同一个标签，则1转移
                    relation_num += 1
                    candidates[b][lj] += predict_vec[li]
                    continue
                # Todo: 两种相关性之间计算一下
                if predict_vec[li] > exists_lower_bound:  # Todo: 不引入阈值了，这样既考虑促进情况，也考虑抑制的情况
                    if requires_grad:
                        a = adjList[li][lj]
                    else:
                        a = adjList[li][lj].item()
                    relation_num += 1
                    # 标签li对标签lj起促进作用，直接相乘即可
                    if a > incr_lower_bound:
                        # 仅当起促进作用时，才累加
                        candidates[b][lj] += predict_vec[li] * a
                    elif a < decr_upper_bound:  # 标签li对lj起抑制作用
                        candidates[b][lj] += 1 - predict_vec[li] * (1 - a)
            # 按照转移的标签数量进行平均
            # 里边既有促进作用的部分，也有抑制作用的部分
            candidates[b][lj] /= relation_num
    return candidates


def LabelOMP(predicts, adjList, label_prob_vec, k=2):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    # 每张图片，找和它最相似的k张图片

    # 最终输出的是图像之间的特征相似度矩阵和语义相似度矩阵
    predict_similarities = torch.zeros(batch_size, batch_size, dtype=torch.float64).to(device)
    # Todo: candidates重复计算了啊
    # candidates表示从一个标签预测向量中根据标签相关性推断出来的新预测向量

    candidates = getCorrectedCandidates(predicts, adjList, label_prob_vec,
                                        requires_grad=False)
    # 对第1维计算范数
    candidate_norms = torch.norm(candidates, dim=1)

    # 遍历每张图片
    for i in range(batch_size):
        # 需要拟合的残差
        predict = predicts[i]
        # 进行k次迭代
        # Todo: 维护相似集合，以及相似图像的特征向量和预测向量
        S = set()
        indexes = []
        candidateX = torch.empty(0, label_dim, dtype=torch.float64).to(device)

        # 现在可以计算内积了
        residual = predict
        predictOnCpu = predict.cpu().numpy()
        for j in range(k):
            candidate_inner_products = torch.matmul(candidates, residual)
            predict_scores = candidate_inner_products / candidate_norms

            # 找到最相似的图像i‘
            # 从中选出相似性最高的图像，加到相似集中
            index = findMaxIndex(predict_scores, S, i)
            # 判断是否满足内积大于等于0的条件
            # 如果不满足，说明找不到相似图片，直接退出即可
            if torch.matmul(predicts[index], residual) <= 0:
                break
            S.add(index)
            indexes.append(index)

            candidateX = torch.cat((candidateX, candidates[index].unsqueeze(0)), dim=0)
            # Todo: 这里不应该预测值拟合，而应该是预测值经相关性的拟合？
            predict_coefficients = my_general_linear_model_func(torch.transpose(candidateX, 0, 1).cpu().numpy(),
                                                                predictOnCpu)
            # 更新相似性矩阵
            # 不是对称的，因此，更新第i行
            for m in range(len(indexes)):
                neighbor = indexes[m]
                # Todo: 验证引入的约束操作对与原解的修改情况
                predict_similarities[i][neighbor] = predict_coefficients[m]  # 确保相似性大于0
                # 更新残差
                residual = predict - torch.matmul(predict_similarities[i], candidates)
    return predict_similarities.detach()  # 对于无需通过pytorch计算图优化的变量，将其detach


class LabelSmoothLoss(nn.Module):
    # 进行相关参数的设置
    def __init__(self, relation_need_grad=False, corrected=False):
        super(LabelSmoothLoss, self).__init__()
        self.relation_need_grad = relation_need_grad
        self.corrected = corrected

    # 传入
    # 1. 预测概率y：b * 80
    # 2. 图像语义相似度: b * b
    # 3. 标签相关性: 80 * 80
    def forward(self, predicts, similarities, adjList, label_prob_vec=None, ):
        device = predicts.device
        batch_size = len(predicts)
        # Todo: 对于每个样本，如果没有相似的图像，则不考虑这个图像的标签平滑损失
        total_loss = 0
        cnt = 0
        candidates = getCorrectedCandidates(predicts, adjList, label_prob_vec, self.relation_need_grad)
        for i in range(batch_size):
            if torch.sum(similarities[i]) == 0:
                continue
            cnt += 1
            total_loss += torch.norm(predicts[i] - torch.matmul(similarities[i], candidates), p=2)
        # Todo: 这里究竟是什么？
        return torch.tensor(0).to(device) if cnt == 0 else total_loss / cnt


# import util
# from util import *
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
        return ap.mean() * 100, ap.tolist()

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


class MyWriter(object):
    def __init__(self, dir_name, stats_name='stats'):
        super(MyWriter, self).__init__()
        self.stats_dir = os.path.join(dir_name, stats_name)
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

    def get(self, file_name, buf_size=1, header=''):
        # 根据文件路径和buffer_size创建文件对象
        file = open(os.path.join(self.stats_dir, file_name), 'w', buffering=buf_size)
        writer = csv.writer(file)
        # 写入表头信息，如果有的话
        if len(header) != 0:
            writer.writerow(header)
        return writer


def gen_adjs(A):
    batch_size = A.size(0)
    adjs = torch.zeros_like(A)
    for i in range(batch_size):
        # 这里对行求和
        D = torch.pow(A[i].sum(1).float(), -0.5)
        # 将其转换成对角矩阵
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A[i], D).t(), D)
        adjs[i] = adj
    return adjs


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
        self.xs_pos = x
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


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    MultiScaleCrop(448, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    Warp(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def construct_relation_by_matrix(num_labels, matrix, device):
    adjList = [dict() for _ in range(num_labels)]
    variables = []
    # Todo: 这里的值到底怎么确定？
    for i in range(num_labels):
        for j in range(num_labels):
            # 自相关性如何处理？额外进行处理
            if i == j:
                continue
            variable = torch.tensor(matrix[i][j]).to(device)
            variable.requires_grad_()
            variables.append(variable)
            # 直接维护对应的优化变量即可
            adjList[i][j] = variable
    return adjList, variables


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', type=str)

parser.add_argument('--batch', default='4', type=int)

parser.add_argument('--k', default='2', type=int)
# 输入不同的学习率
parser.add_argument('--lr', default='0.0001', type=float)

parser.add_argument('--rlr', default='0.0001', type=float)

parser.add_argument('--predict_gap', default='0', type=float)

parser.add_argument('--relation_gap', default='0', type=float)

parser.add_argument('--labmda_y', default='1.0', type=float)

parser.add_argument('--env', default='client', type=str)

args = parser.parse_args()

env = args.env

if env == 'server':
    root_path = "/data/projects/dataset/voc_standalone"
    category_dir = '/data/projects/fate/my_practice/dataset/voc_expanded'
    json_file = '/data/projects/fate/my_practice/dataset/voc_expanded/old_image_ids/train_image_id.json'
else:
    root_path = "/home/klaus125/research/dataset/voc_standalone"
    category_dir = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded'
    json_file = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded/old_image_ids/train_image_id.json'

num_labels = 20

# 既定的配置，一般不会发生变化
num_workers = 4
drop_last = True

device = args.device
lr = args.lr
rlr = args.rlr
k = args.k
batch_size = args.batch

predict_gap = args.predict_gap
relation_gap = args.relation_gap
lambda_y = args.labmda_y
# Todo: 设置设备id？

train_dataset = Voc2007Classification(root_path, "trainval", config_dir=category_dir)
train_dataset.transform = train_transforms
valid_dataset = Voc2007Classification(root_path, "test", config_dir=category_dir)
valid_dataset.transform = val_transforms

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
    drop_last=drop_last, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
    drop_last=drop_last, shuffle=False
)

stats_dir = f'{method}_{batch_size}_{k}_{lr}_{rlr}_{predict_gap}_{relation_gap}_{lambda_y}_stats'

my_writer = MyWriter(dir_name=os.getcwd(), stats_name=stats_dir)

train_writer = my_writer.get("train.csv", header=['epoch', 'mAP', 'entropy_loss', 'relation_loss', 'overall_loss'])
valid_writer = my_writer.get("valid.csv", header=['epoch', 'mAP'])

train_aps_writer = my_writer.get("train_aps.csv")
val_aps_writer = my_writer.get("val_aps.csv")

# 使用resnet-101模型
model = torch_models.resnet101(pretrained=True, num_classes=1000)
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_labels))
torch.nn.init.kaiming_normal_(model.fc[0].weight.data)
model = model.to(device)

criterion = AsymmetricLossOptimized().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

ap_meter = AveragePrecisionMeter(difficult_examples=True)

# Todo: 从标签共现文件中计算出邻接矩阵
image_id2labels = json.load(open(json_file, 'r'))

# 计算正向的标签共现概率矩阵和概率矩阵
adjMatrix = np.zeros((num_labels, num_labels))
nums = np.zeros(num_labels)
for image_info in image_id2labels:
    labels = image_id2labels[image_info]['labels']
    for label in labels:
        nums[label] += 1
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            x = labels[i]
            y = labels[j]
            adjMatrix[x][y] += 1
            adjMatrix[y][x] += 1
label_prob_vec = nums / len(image_id2labels)
nums = nums[:, np.newaxis]
# 遍历每一行
for i in range(num_labels):
    if nums[i] != 0:
        adjMatrix[i] = adjMatrix[i] / nums[i]
# 遍历A，将主对角线元素设置为1
for i in range(num_labels):
    adjMatrix[i][i] = 1

# variables中既包含了正向变量，也包含负向变量
adjList, variables = construct_relation_by_matrix(num_labels, adjMatrix, device)

relation_optimizer = torch.optim.SGD(variables, lr=rlr)
# Todo: 更改，40或者20
epochs = 200
# threshold = 0.2


# 开始训练
for epoch in range(epochs):

    # 训练
    ap_meter.reset()
    model.train()

    # 训练阶段
    total_samples = len(train_loader.sampler)
    steps_per_epoch = math.ceil(total_samples / batch_size)

    ENTROPY_LOSS_KEY = 'Entropy Loss'
    RELATION_LOSS_KEY = 'Relation Loss'
    OVERALL_LOSS_KEY = 'Overall Loss'
    losses = OrderedDict([(ENTROPY_LOSS_KEY, tnt.AverageValueMeter()),
                          (RELATION_LOSS_KEY, tnt.AverageValueMeter()),
                          (OVERALL_LOSS_KEY, tnt.AverageValueMeter())])

    # 直接输出信息到控制台上
    for train_step, (features, target) in enumerate(train_loader):
        features = features.to(device)

        # Todo: 注意这里的数据转换部分
        prev_target = target.clone()
        # 对target进行转换
        target[target == 0] = 1
        target[target == -1] = 0
        target = target.to(device)

        output = model(features)

        predicts = torch.sigmoid(output).to(torch.float64)

        ap_meter.add(predicts.data, prev_target)

        predict_similarities = LabelOMP(predicts.detach(), adjList, label_prob_vec, k)

        label_loss = LabelSmoothLoss(relation_need_grad=True, corrected=True)(predicts.detach(),
                                                                              predict_similarities,
                                                                              adjList, label_prob_vec)

        # 需要先对label_loss进行反向传播，梯度下降更新标签相关性
        # 如果标签平滑损失不为0，才进行优化
        if label_loss != 0:
            relation_optimizer.zero_grad()
            label_loss.backward()
            relation_optimizer.step()
        # 确保标签相关性在0到1之间

        # 遍历每个优化变量，对其值进行约束，限定在[0,1]之内
        for variable in variables:
            variable.data = torch.clamp(variable.data, min=0.0, max=1.0)

        # 总损失 = 交叉熵损失 + 标签相关性损失
        # 优化CNN参数时，need_grad设置为True，表示需要梯度

        entropy_loss = criterion(predicts, target)

        lambda_y = 1

        relation_loss = lambda_y * \
                        LabelSmoothLoss(relation_need_grad=False, corrected=True)(predicts,
                                                                                  predict_similarities,
                                                                                  adjList,
                                                                                  label_prob_vec
                                                                                  )

        # 初始化标签相关性，先计算标签平滑损失，对相关性进行梯度下降
        loss = entropy_loss + relation_loss

        losses[OVERALL_LOSS_KEY].add(loss.item())
        losses[ENTROPY_LOSS_KEY].add(entropy_loss.item())
        losses[RELATION_LOSS_KEY].add(relation_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Todo: 模型参数的学习率下降
    if (epoch + 1) % 4 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9

    mAP, aps = ap_meter.value()

    train_writer.writerow(
        [epoch, mAP.item(), losses[ENTROPY_LOSS_KEY].mean, losses[RELATION_LOSS_KEY].mean,
         losses[OVERALL_LOSS_KEY].mean])
    train_aps_writer.writerow(aps)

    # 验证阶段
    model.eval()
    ap_meter.reset()

    total_val_samples = len(val_loader.sampler)
    val_steps_per_epoch = math.ceil(total_samples / batch_size)
    epoch_val_loss = 0
    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):
            features = features.to(device)

            prev_target = target.clone()
            # Todo: voc中标签特点
            #  0  ： 存在但难以识别
            #  -1 ： 不存在
            #  1  :  存在
            #  当前处理方法，将0设置为存在，并在计算指标时跳过对应的标签
            target[target == 0] = 1
            target[target == -1] = 0
            target = target.to(device)

            output = model(features)
            predicts = torch.sigmoid(output)

            ap_meter.add(predicts.data, prev_target)

        mAP, aps = ap_meter.value()
        valid_writer.writerow([epoch, mAP.item()])
        val_aps_writer.writerow(aps)
