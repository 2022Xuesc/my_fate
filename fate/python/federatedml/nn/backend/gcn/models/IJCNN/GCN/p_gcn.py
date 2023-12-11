from torch.nn import Parameter

from federatedml.nn.backend.gcn.models.graph_convolution import GraphConvolution
from federatedml.nn.backend.gcn.utils import *


class ResnetPGCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, out_channels=2048, adjList=None):
        super(ResnetPGCN, self).__init__()
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

        # Todo: 这里还要加一个全连接层
        self.fc = nn.Linear(in_features=in_channel, out_features=num_classes)
        torch.nn.init.kaiming_normal_(self.fc.weight.data)

        # 定义图卷积层
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: 生成邻接的相关信息？
        self.A = self.generateA(adjList)

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

        # 经过gcn得到预测
        y1 = self.fc(feature)

        # feature的形状是batch_size * feature_dim
        # 将其与self.fc的权重进行哈达玛积
        # 得到batch_size * feature_dim * num_classes的输入
        gcn_input = (feature.unsqueeze(2).expand(-1, -1, self.num_classes).transpose(1,
                                                                                     2) * self.fc.weight)  # 执行完transpose后形状为(1,80,2048)，每一行是1个图像特征

        # Todo: 是图神经网络的输入

        adj = gen_adj(self.A).detach()
        x = self.gc1(gcn_input, adj)
        x = self.relu(x)  # 需要经过leakyReLU
        x = self.gc2(x, adj)
        # 此时x是一个batch_size * num_classes * 1的向量
        y2 = x.view((x.size(0), -1))
        return y1 + y2

    # 为不同部分设置不同的学习率
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            # 注意全连接层的参数也要进行优化
            {'params': self.fc.parameters(), 'lr': lr}
        ]

    def get_feature_params(self):
        return self.features.parameters()

    def get_gcn_params(self):
        return [
            {'params': self.gc1.parameters()},
            {'params': self.gc2.parameters()}
        ]
