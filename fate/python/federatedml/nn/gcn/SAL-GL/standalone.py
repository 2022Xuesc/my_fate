# 服务器与客户端的通用逻辑
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.multi_label.losses.SmoothLoss import *

import json
import os
import pickle
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader


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
        non_zero_labels = 0
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            if targets.sum() != 0:
                non_zero_labels += 1
        # Todo: 在这里判断不为空的标签个数，直接求均值
        return ap.sum() / non_zero_labels

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
        if pos_count != 0:
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


# train_path = '/data/projects/iid_dataset/client1/train'
# category_dir = '/data/projects/dataset'
train_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'
inp_name = 'coco_glove_word2vec.pkl'
dataset_loader = DatasetLoader(category_dir, train_path, train_path, inp_name)

device = 'cuda:0'
batch_size = 4
train_loader, valid_loader = dataset_loader.get_loaders(batch_size)


# 场景概率的熵损失
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


# 低秩双线性注意力
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


import torch.nn.functional as F
from torch.nn import Parameter


# 图卷积层
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

    def reset_parameters(self):
        # Todo: 这里self.weight.size(1)表示输出维度out_features
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

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


# 给定邻接表矩阵对应的Parameter参数，进行图卷积的等价处理
# Todo: 这里要仔细推导一下
def gen_adj(A):
    # 分批次进行处理
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


# 定义SAL-GL模型
class SALGL(nn.Module):
    # 传进来一个resnet101，截取其前一部分作为主干网络
    def __init__(self, model, feat_dim=2048, att_dim=1024, num_scenes=2, in_channels=300,
                 out_channels=2048, num_classes=80, comat_smooth=False):

        super(SALGL, self).__init__()
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
        self.comat_smooth = comat_smooth

        # 使用自定义的MaxPooling
        self.pooling = nn.MaxPool2d(14, 14)
        # 计算场景的线性分类器
        self.scene_linear = nn.Linear(feat_dim, num_scenes, bias=False)
        # 图卷积网络部分
        self.gc1 = GraphConvolution(in_channels, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: register_buffer的好处
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        # 计算最大熵，预测向量呈均匀分布时的熵
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes).cuda())

        # embed_dim是标签嵌入的维度，300维？
        # embed_dim = 300
        self.attention = LowRankBilinearAttention(feat_dim, in_channels, att_dim)

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
        img_feats = self.pooling(img_feats)
        # Todo: 将pooling后的特征展平
        img_feats = img_feats.view(img_feats.size(0), -1)

        scene_scores = self.scene_linear(img_feats)
        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            # y的维度是(B, num_labels)
            # batch_comats的维度是(B, num_labels, num_labels) 是0-1矩阵，如果为1，说明对应标签对同时出现了
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
            # _scene_probs的维度是(B, num_scenes)
            for i in range(_scene_probs.shape[0]):
                # Todo: 采用各个场景的标签共现矩阵加权还是说只取最大概率场景的标签共现矩阵
                if self.comat_smooth:
                    prob = _scene_probs[i].unsqueeze(-1).unsqueeze(-1)  # [num_scenes, 1, 1]
                    comat = batch_comats[i].unsqueeze(0)  # [1, num_labels, num_labels]
                    # 当前comat对每个场景都有贡献，贡献值即为scene的预测概率值
                    self.comatrix += prob * comat  # [num_scenes, num_labels, num_labels]
                else:
                    max_scene_id = torch.argmax(_scene_probs[i])
                    # comats只加到对应的最大概率场景的标签共现矩阵上
                    self.comatrix[max_scene_id] += batch_comats[i]
        # 计算场景的概率
        scene_probs = F.softmax(scene_scores, dim=-1)
        # 计算场景概率的熵损失
        sample_en = self.entropy(scene_probs)
        # 计算该批次场景预测向量的平均值
        _scene_probs = torch.mean(scene_probs, dim=0)
        # 计算和最大熵的差距，应该使这种差距尽可能小，使场景的预测更加多样化，缓解winner-take-all的问题
        # 这个是批次平均概率向量的熵，而前面的是概率向量熵的平均值，两者代表的是不同的含义
        batch_en = (self.max_en - self.entropy(_scene_probs)) * 100

        # attention计算每个标签的视觉表示
        # Todo: Transformer挖掘空间特征，复习Transformer
        # label_feats, alphas = self.attention(img_feats, inp)

        # 根据场景预测向量选择合适的标签共现矩阵
        if not self.comat_smooth:
            comats = self.comatrix2prob()
            # 取出每张图像对应的概率最大的场景索引
            indices = torch.argmax(scene_probs, dim=-1)
            # 取出对应的标签概率矩阵
            comats = torch.index_select(comats, dim=0, index=indices)
        else:  # 否则的话，进行一下加权
            _scene_probs = scene_probs.unsqueeze(-1).unsqueeze(-1)  # [bs,num_scenes,1,1]
            comats = self.comatrix2prob().unsqueeze(0)  # [1,num_scenes,nc,nc]
            comats = _scene_probs * comats  # [bs,num_scenes,nc,nc]
            # 根据场景预测概率向量对每个场景的标签概率矩阵进行加权，即soft策略
            comats = torch.sum(comats, dim=1)  # [bs,nc,nc]

        # 输入标签的视觉特征和共现矩阵
        # label_feats的维度是[batch_size, num_labels, att_dim]
        # comats的维度是[batch_size, num_labels, num_labels]
        # 与传统的GCN相比多了一个batch维度
        # 1. 先对comats进行adj的转换
        adjs = gen_adj(comats)
        # 每张图片采用不同的邻接矩阵
        x = self.gc1(inp, adjs)
        x = self.relu(x)
        x = self.gc2(x, adjs)  # [batch_size, num_classes, feat_dim]

        # 此时x的输出就是每张图像的分类器
        output = torch.matmul(x, img_feats.unsqueeze(2)).squeeze(-1)
        # Todo: 求点积，得到预测向量
        # 返回前向传播的结果
        return {
            'output': output,
            'scene_probs': scene_probs,
            'sample_en': sample_en,
            'batch_en': batch_en,
            # 'att_weights': alphas,
            'comat': comats
        }


model = torch_models.resnet101(pretrained=True, num_classes=1000)

model = SALGL(model).to(device)

ap_meter = AveragePrecisionMeter()

model.train()

sigmoid_func = torch.nn.Sigmoid()
criterion = AsymmetricLossOptimized().to(device)

for epoch in range(100):
    for train_step, ((features, inp), target) in enumerate(train_loader):
        features = features.to(device)
        target = target.to(device)
        inp = inp.to(device)
        output = model(features, inp, y=target)

        logits = sigmoid_func(output['output'])
        # Todo: 这里需要考虑多种损失函数
        cross_entropy_loss = criterion(logits, target)

        # 还有非对称损失
        loss = cross_entropy_loss + output['sample_en'] + output['batch_en']
        print(loss.item())
        loss.backward()
        # 优化器步进
