import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Parameter

from federatedml.nn.backend.gcn.models.graph_convolution import GraphConvolution
from federatedml.nn.backend.gcn.models.salgl import EntropyLoss, LowRankBilinearAttention
from federatedml.nn.backend.gcn.utils import gen_adjs, gen_adj
import math
from torch import Tensor
from typing import Optional
from federatedml.nn.backend.gcn.kmeans_torch import kmeans

import copy


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


# 定义SAL-GL模型
# 对于每个网络层，都进行权重初始化
#  1. self.features 使用预训练权重
#  2. nn.Linear 已经进行了权重初始化-->恺明初始化
#  3. 自定义的Parameter也已经进行了权重初始化
class ResnetKmeans(nn.Module):
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

        super(ResnetKmeans, self).__init__()
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
        self.pooling = nn.MaxPool2d(14, 14)
        # 维护场景中心，是一个num_scenes,feat_dim维的tensor
        self.centers = torch.Tensor(num_scenes, feat_dim)
        # 对场景中心初始化，使其分布均匀
        # 那也要让传入的图像上下文特征尽量分布均匀
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

        self.total_scene_cnts = [0] * num_scenes

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)

        # 图卷积网络部分
        self.gc1 = GraphConvolution(feat_dim, gcn_middle_dim)
        self.gc2 = GraphConvolution(gcn_middle_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: 这里的是标签共现矩阵，标签相关性矩阵实时计算
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

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
        att_feats = torch.flatten(img_feats, start_dim=2).transpose(1, 2)
        # Todo: 这里使用MaxPooling还是AveragePooling？
        feature = self.pooling(img_feats)

        feature = feature.view(feature.size(0), -1)

        scene_ids_x, self.centers = kmeans(feature, num_clusters=self.num_scenes, initial_state=self.centers,
                                           scene_cnts=self.total_scene_cnts)

        if self.training:
            # y的维度是(B, num_labels)
            # batch_comats的维度是(B, num_labels, num_labels) 是0-1矩阵，如果为1，说明对应标签对同时出现了
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
            # _scene_probs的维度是(B, num_scenes)
            for i in range(len(scene_ids_x)):
                max_scene_id = scene_ids_x[i]
                # comats只加到对应的最大概率场景的标签共现矩阵上
                self.comatrix[max_scene_id] += batch_comats[i]
                # 更新scene_cnts
                self.total_scene_cnts[max_scene_id] += 1

        # attention计算每个标签的视觉表示
        # attention: [batch_size]
        label_feats, alphas = self.attention(att_feats, inp)

        comats = self.comatrix2prob()
        # 取出每张图像对应的概率最大的场景索引

        # 取出对应的标签概率矩阵
        comats = torch.index_select(comats, dim=0, index=scene_ids_x)
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
            'comat': comats
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
            {'params': self.classifier.parameters(), 'lr': lr}
        ]
