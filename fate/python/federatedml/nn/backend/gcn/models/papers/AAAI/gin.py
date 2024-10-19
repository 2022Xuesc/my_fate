import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# 定义GIN层
class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mid_features = out_features * 2

        self.fc1 = torch.nn.Linear(self.in_features, self.mid_features, bias)
        self.fc2 = torch.nn.Linear(self.mid_features, self.out_features, bias)

        self.bn = nn.BatchNorm1d(self.out_features)

        # epsilon初始化为0，需要进行优化
        self.epsilon = Parameter(torch.zeros(1), requires_grad=True)

    # 前向传播，权重与输入相乘、结果再与邻接矩阵adj相乘。
    def forward(self, x, adj):
        # adj在左边，adj是转置后的[a11,a21,a31...]
        x = (1 + self.epsilon) * x + torch.matmul(adj, x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x


class AAAI_GIN(nn.Module):
    def __init__(self, model, num_classes, in_channels=300, out_channels=1024,
                 latent_dim=512, adjList=None, inp=None):
        super(AAAI_GIN, self).__init__()
        self.A = self.generateA(adjList)  
        # 定义特征提取部分的网络
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

        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 定义图同构层
        self.gin1 = GINLayer(in_channels, out_channels)
        self.gin2 = GINLayer(out_channels, out_channels)
        self.inp = inp
        # 定义重要性向量层
        # Todo: 考虑对这里的层进行修改
        self.imp_layer = nn.Linear(out_channels, latent_dim, True)
        self.latent_layer = nn.Linear(2048, latent_dim, True)
        self.map_layer = nn.Linear(latent_dim, latent_dim, True)

        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        # 最后全连接分类器层
        self.classifier = nn.Linear(latent_dim, 1, True)

    def generateA(self, adjList):
        adj = torch.from_numpy(adjList.astype(np.float32))
        adj = torch.transpose(adj, 0, 1)
        return Parameter(adj)

    def updateA(self, adjList):
        adj = torch.from_numpy(adjList).float()
        adj = torch.transpose(adj, 0, 1)
        self.A.data.copy_(adj)

    # 前向传播逻辑：features --> pooling -->
    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0] if self.inp is None else self.inp  # 从数据集中获取初始标签嵌入
        x = self.gin1(inp, self.A)
        x = self.relu(x)

        identity = x  # 添加残差连接
        x = self.gin2(x, self.A)

        x = self.relu(x + identity)

        # Todo: 根据self.A和x计算pairwise_loss
        m = x / torch.norm(x, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
        similarity = torch.mm(m, m.T)

        diff = (similarity - (self.A + 1))
        pairwise_loss = torch.sum(diff * diff) / (self.num_classes * self.num_classes)

        imp_y = self.sigmoid(self.imp_layer(x))

        latent_feature = self.relu(self.latent_layer(feature))
        # 哈达玛积
        # batch_size * num_labels * latent_dim
        x = torch.mul(imp_y, torch.unsqueeze(latent_feature, dim=1))

        x = self.relu(self.map_layer(x))

        x = self.classifier(x).squeeze(2)

        return x, pairwise_loss

    # Todo: 下面获取参数的方法都还要进行修改

    def get_feature_params(self):
        return self.features.parameters()

    # 为不同部分设置不同的学习率
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gin1.parameters(), 'lr': lr},
            {'params': self.gin2.parameters(), 'lr': lr},
            {'params': self.imp_layer.parameters(), 'lr': lr},
            {'params': self.latent_layer.parameters(), 'lr': lr},
            {'params': self.map_layer.parameters(), 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr},
        ]
