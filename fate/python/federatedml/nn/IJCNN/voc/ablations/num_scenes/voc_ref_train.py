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

import torch.backends.cudnn as cudnn
import math
import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import Parameter

import torch.nn.functional as F

import csv
import json
import os
import os.path
import pickle
import random


class MyWriter(object):
    def __init__(self, dir_name):
        super(MyWriter, self).__init__()
        self.stats_dir = os.path.join(dir_name, 'stats')
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

    def get(self, file_name, buf_size=1, header=''):
        # 根据文件路径和buffer_size创建文件对象
        file = open(os.path.join(self.stats_dir, file_name), 'w', buffering=buf_size)
        writer = csv.writer(file)
        # 写入表头信息，如果有的话
        writer.writerow(header)
        return writer


my_writer = MyWriter(dir_name=os.getcwd())

train_writer = my_writer.get("train.csv", header=['epoch', 'mAP', 'train_loss'])
valid_writer = my_writer.get("valid.csv", header=['epoch', 'mAP', 'valid_loss'])

train_aps_writer = my_writer.get("train_aps.csv")
val_aps_writer = my_writer.get("val_aps.csv")


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


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


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


class Voc2007Classification(Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, ):
        self.root = root
        self.set = set
        self.path_images = os.path.join(root, set)
        self.transform = transform
        self.target_transform = target_transform

        path_csv = os.path.join(self.root, 'csvs')
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


class LowRankBilinearAttention(nn.Module):
    def __init__(self, dim1, dim2, att_dim=2048):
        """

        Parameters
        ----------
        dim1: 图像的特征大小
        dim2: 标签的特征大小
        att_dim: 注意力网络的大小
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)
        self.hidden_linear = nn.Linear(att_dim, att_dim)
        self.target_linear = nn.Linear(att_dim, 1)
        # 双曲正切函数
        self.tanh = nn.Tanh()
        # 使用softmax层来计算权重
        # 标签维在前，像素位置维在后，因此，对像素位置维进行softmax，得到关于每个标签的像素位置维权重，即空间注意力值
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        执行前向传播
        Parameters
        ----------
        x1: 维度(B, num_pixels, dim1)，每个像素点都有一个特征向量，dim1应该就是通道吧？
        x2: 维度(B, num_labels, dim2)，每个标签都有一个特征向量，每个批次的标签向量都是一样的？
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  # (B, 1, num_pixels, att_dim)
        _x2 = self.linear2(x2).unsqueeze(dim=2)  # (B, num_labels, 1, att_dim)
        # 这样点乘之后，可以得到B,num_labels, num_pixels, att_dim
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        # 将向量计算输出logit，消除掉最后一维，得到(B, num_labels, num_pixels)
        t = self.target_linear(t).squeeze(-1)
        # 经过softmax，归一化，得到每个标签的空间位置权重
        alpha = self.softmax(t)
        label_repr = torch.bmm(alpha, x1)
        # 返回标签的特征表示（每个空间位置的加权）以及注意力值
        return label_repr, alpha


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重即为从输入特征数到输出特征数的一个变换矩阵
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            # 这里注意偏置的维度含义：batch？channel？
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            # 声明不具有偏置项
            self.register_parameter('bias', None)
        # 这里的reset方法是自定义的
        self.reset_parameters()
        # self.init_parameters_by_kaiming()

    def reset_parameters(self):
        # Todo: 这里self.weight.size(1)表示输出维度out_features
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 进行恺明初始化
    def init_parameters_by_kaiming(self):
        torch.nn.init.kaiming_normal_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.kaiming_normal_(self.bias.data)

    # 前向传播，权重与输入相乘、结果再与邻接矩阵adj相乘。
    # adj * input * weight
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        # Todo: 这里是否需要加batch norm？
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


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


class EntropyLoss(nn.Module):
    def __init__(self, margin=0.0) -> None:
        super().__init__()
        self.margin = margin

    # 前向传播
    # 计算概率向量的熵
    def forward(self, x, eps=1e-7):
        # 计算xlogx
        x = x * torch.log(x + eps)
        # 对每个批次进行求和，计算熵
        en = -1 * torch.sum(x, dim=-1)
        # 求均值，该batch的预测向量的熵的平均值
        # 越小说明每张图像对场景的预测越准确，越好
        en = torch.mean(en)
        return en


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class ResnetSalgl(nn.Module):
    # 传进来一个resnet101，截取其前一部分作为主干网络
    # Todo: 比较重要的超参数
    #  1. 场景个数 num_scenes=4
    #  3. 低秩双线性注意力中联合嵌入空间的维度：att_dim= 1024(300维和2048计算相关性)
    #  4. 图卷积神经网络部分，中间层的维度：gcn_middle_dim=1024
    #                     最后输出层的维度out_channels要和feat_dim相等，以进行concat操作
    #  5. 是否进行相关性矩阵平滑：comat_smooth=False
    #  
    def __init__(self, model, img_size=448, embed_dim=300, feat_dim=2048, att_dim=1024, num_scenes=6,
                 gcn_middle_dim=1024, out_channels=2048, num_classes=80):

        super(ResnetSalgl, self).__init__()
        # cnn主干网络，提取特征
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_scenes = num_scenes
        self.num_classes = num_classes

        self.total_scene_cnts = [0] * num_scenes
        self.scene_linear = nn.Linear(feat_dim, num_scenes, bias=False)

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)

        # 图卷积网络部分
        self.gc1 = GraphConvolution(feat_dim, gcn_middle_dim)
        self.gc2 = GraphConvolution(gcn_middle_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: 这里的是标签共现矩阵，标签相关性矩阵实时计算
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes))

        # embed_dim = 300
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(num_classes, feat_dim)

    # 将(对称的)共现矩阵转换成(非对称的)概率矩阵
    def comatrix2prob(self):
        # comat: [num_scenes,nc,nc]
        # self.comatrix在训练时更新
        # Todo: 这里先不进行转置，pij仍然表示标签i出现的条件下标签j出现的
        # comat = torch.transpose(self.comatrix, dim0=1, dim1=2)
        comat = self.comatrix
        # 计算每个场景中每个标签的出现次数，作为分母
        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)
        # 标签共现矩阵除以每个标签的出现次数，即为概率矩阵
        comat = comat / (temp + 1e-8)
        # 主对角线部分设置为1
        for i in range(self.num_classes):
            comat[:, i, i] = 1
        # 引入自连接
        return comat

    """
    x: 图像特征
    y: 图像标签集合
    inp: 标签嵌入
    """

    def forward(self, x, inp, y=None):
        img_feats = self.features(x)
        img_feats = torch.flatten(img_feats, start_dim=2).transpose(1, 2)
        # Todo: 这里使用MaxPooling还是AveragePooling？
        # feature = self.pooling(img_feats)

        # feature = feature.view(feature.size(0), -1)
        img_contexts = torch.mean(img_feats, dim=1)

        scene_scores = self.scene_linear(img_contexts)

        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            # y的维度是(B, num_labels)
            # batch_comats的维度是(B, num_labels, num_labels) 是0-1矩阵，如果为1，说明对应标签对同时出现了
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
            # _scene_probs的维度是(B, num_scenes)
            for i in range(_scene_probs.shape[0]):
                max_scene_id = torch.argmax(_scene_probs[i])
                # comats只加到对应的最大概率场景的标签共现矩阵上
                self.comatrix[max_scene_id] += batch_comats[i]
                # 更新scene_cnts
                self.total_scene_cnts[max_scene_id] += 1

        scene_probs = F.softmax(scene_scores, dim=-1)
        # 计算场景概率的熵损失
        sample_en = self.entropy(scene_probs)
        # 计算该批次场景预测向量的平均值
        _scene_probs = torch.mean(scene_probs, dim=0)
        batch_en = self.max_en - self.entropy(_scene_probs)
        # attention计算每个标签的视觉表示
        # attention: [batch_size]
        label_feats, alphas = self.attention(img_feats, inp)

        comats = self.comatrix2prob()
        # 取出每张图像对应的概率最大的场景索引
        indices = torch.argmax(scene_probs, dim=-1)
        comats = torch.index_select(comats, dim=0, index=indices)
        # 输入标签的视觉特征和共现矩阵
        # label_feats的维度是[batch_size, num_labels, att_dim]
        # comats的维度是[batch_size, num_labels, num_labels]
        # 与传统的GCN相比多了一个batch维度
        # 1. 先对comats进行adj的转换
        adjs = gen_adjs(comats)
        # 每张图片采用不同的邻接矩阵
        x = self.gc1(label_feats, adjs)
        x = self.relu(x)
        x = self.gc2(x, adjs)  # [batch_size, num_classes, feat_dim]

        output = torch.cat([label_feats, x], dim=-1)
        output = self.fc(output)
        output = self.classifier(output)

        # Todo: 求点积，得到预测向量
        # 返回前向传播的结果
        return {
            'output': output,
            'scene_probs': scene_probs,
            'entropy_loss': sample_en + batch_en,
            'comat': comats,
            'scene_indices': indices
        }

    # Todo: 获取需要优化的参数
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.attention.parameters(), 'lr': lr},
            {'params': self.fc.parameters(), 'lr': lr},
            # Todo: Debug查看一下分类器参数
            {'params': self.classifier.parameters(), 'lr': lr},
            # Todo: 以下是不能聚合的参数
            {'params': self.scene_linear.parameters(), 'lr': lr}
        ]


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

# batch_size = 2
# root_path = '/home/klaus125/research/dataset/voc_standalone'
# device = 'cuda:0'
# epoch_step = 1

batch_size = 16
root_path = "/data/projects/voc_standalone"
# epoch_step = 30

num_labels = 20
momentum = 0.9
# 相关训练参数的设置，和client1的配置相一致
# lr = 0.1
# lrp = 0.1
learning_rate = 1e-4

# 既定的配置，一般不会发生变化
num_workers = 32
drop_last = False
pin_memory = True
cudnn.benchmark = True

# Todo: 设置设备id？

train_dataset = Voc2007Classification(root_path, "trainval")
train_dataset.transform = train_transforms
valid_dataset = Voc2007Classification(root_path, "test")
valid_dataset.transform = val_transforms

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
    drop_last=drop_last, pin_memory=pin_memory, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
    drop_last=drop_last, pin_memory=pin_memory, shuffle=False
)

# res_model = torch_models.resnet101(pretrained=True)
# model = MyResNetModel(model=res_model, num_classes=num_labels).to(device)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_scenes', default=4, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
# 输入不同的学习率
parser.add_argument('--lr', default='0.0001',type=float)
parser.add_argument("--lrp", default='0.1',type=float)

args = parser.parse_args()

num_scenes = args.num_scenes
device = args.device
lr, lrp = args.lr, args.lrp



model = torch_models.resnet101(pretrained=False, num_classes=1000)

# 使用salgl模型
model = ResnetSalgl(model,num_scenes=num_scenes,num_classes=num_labels).to(device)

criterion = AsymmetricLossOptimized().to(device)


optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp), lr=lr, weight_decay=1e-4)


ap_meter = AveragePrecisionMeter(difficult_examples=True)

# Todo: 更改，40或者20
epochs = 200

# 开始训练
for epoch in range(epochs):
    scene_images = dict()
    for i in range(num_scenes):
        scene_images[i] = []
    ap_meter.reset()
    model.train()
    
    # 训练阶段
    total_samples = len(train_loader.sampler)
    steps_per_epoch = math.ceil(total_samples / batch_size)

    # 直接输出信息到控制台上
    for train_step, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        # Todo: 注意这里的数据转换部分
        prev_target = target.clone()
        # 对target进行转换
        target[target == 0] = 1
        target[target == -1] = 0
        target = target.to(device)

        output = model(inputs)
        ap_meter.add(output.data, prev_target)

        # Todo: 这里x不需要经过sigmoid
        loss = criterion(torch.nn.Sigmoid()(output), target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= steps_per_epoch
    # after_scheduler_lr = optimizer.param_groups[0]['lr']

    # Todo: 更新学习率
    # if (epoch + 1) % epoch_step == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * 0.1

    mAP, aps = ap_meter.value()

    train_writer.writerow([epoch, mAP.item(), epoch_loss])
    train_aps_writer.writerow(aps)

    # 验证阶段
    model.eval()
    ap_meter.reset()

    total_val_samples = len(val_loader.sampler)
    val_steps_per_epoch = math.ceil(total_samples / batch_size)
    epoch_val_loss = 0
    for i, (inputs, target) in enumerate(val_loader):
        inputs = inputs.to(device)
        prev_target = target.clone()
        target[target == 0] = 1
        target[target == -1] = 0
        target = target.to(device)

        output = model(inputs)
        loss = criterion(torch.nn.Sigmoid()(output), target)
        epoch_val_loss += loss.item()
        ap_meter.add(output.data, prev_target)
    mAP, aps = ap_meter.value()
    epoch_val_loss /= val_steps_per_epoch
    valid_writer.writerow([epoch, mAP.item(), epoch_val_loss])
    val_aps_writer.writerow(aps)
