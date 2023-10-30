import torchvision.models as torch_models
from torch.nn import Parameter

from federatedml.nn.backend.gcn.utils import *


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重即为从输入特征数到输出特征数的一个变换矩阵
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        # self.bn1 = nn.BatchNorm1d(out_features)
        # self.relu = nn.LeakyReLU(negative_slope=0.2)

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
        # 加一个batch_norm层
        # support = self.bn1(support)
        # support = self.relu(support)
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


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, out_channels=2048, t=0, adjList=None):
        super(GCNResnet, self).__init__()
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
        # self.fc = nn.Linear(in_features=2048,out_features=80)
        # #
        # torch.nn.init.kaiming_normal_(self.fc.weight.data)

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


class PGCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, out_channels=2048, adjList=None):
        super(PGCNResnet, self).__init__()
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
        gcn_input = (feature.unsqueeze(2).expand(-1, -1, self.num_classes).transpose(1, 2) * self.fc.weight)  # 执行完transpose后形状为(1,80,2048)，每一行是1个图像特征

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
        ]

    def get_feature_params(self):
        return self.features.parameters()

    def get_gcn_params(self):
        return [
            {'params': self.gc1.parameters()},
            {'params': self.gc2.parameters()}
        ]


def gcn_resnet101(pretrained, dataset, t, adjList=None, device='cpu', num_classes=80, in_channel=300):
    model = torch_models.resnet101(pretrained=pretrained)

    model = GCNResnet(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adjList=adjList)
    return model.to(device)


def p_gcn_resnet101(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=2048, out_channel=1):
    model = torch_models.resnet101(pretrained=pretrained)
    model = PGCNResnet(model=model, num_classes=num_classes, in_channel=in_channel, out_channels=out_channel,
                       adjList=adjList)
    return model.to(device)
