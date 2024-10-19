from torch.nn import Parameter

from federatedml.nn.backend.gcn.models.graph_convolution import GraphConvolution
from federatedml.nn.backend.gcn.utils import *


class ResnetCGCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, out_channels=2048, t=0, adjList=None):
        super(ResnetCGCN, self).__init__()
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
        self.pooling = nn.MaxPool2d(14, 14)  # Todo: 这里使用MaxPool2d好还是AdaptivePool好？之后进行验证

        # 定义图卷积层
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        # self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: 生成邻接的相关信息？
        self.A = self.generateA(adjList)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def generateA(self, adjList):
        return Parameter(torch.from_numpy(adjList).float())

    def updateA(self, adjList):
        self.A.data.copy_(torch.from_numpy(adjList).float())

    # 前向传播逻辑：features --> pooling -->
    def forward(self, feature, inp, _adj=None):
        feature = self.features(feature)
        feature = self.pooling(feature)
        # Todo: 第一维应该是batch，这里将二维特征展平
        feature = feature.view(feature.size(0), -1)

        # Todo: inp是图神经网络的输入
        # 这里选择取第一个分量，是因为每个数据样本使用的inp都是相同的
        inp = inp[0]
        adj = gen_adj(self.A).detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        # Todo: 添加全连接层进行比较
        # x = self.fc(feature)
        return x

    # 为不同部分设置不同的学习率
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]

    def get_feature_params(self):
        return self.features.parameters()

    def get_gcn_params(self):
        return [
            {'params': self.gc1.parameters()},
            {'params': self.gc2.parameters()}
        ]
