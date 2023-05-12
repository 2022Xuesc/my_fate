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
        if bias:
            # 这里注意偏置的维度含义：batch？channel？
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
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

    # 前向传播，权重与输入相乘、结果再与邻接矩阵adj相乘。
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
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
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)
        # 定义图卷积层
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: 生成邻接的相关信息？
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    # 前向传播逻辑：features --> pooling -->
    def forward(self, feature, inp):
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
        return x

    # 获取需要优化的参数
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]


def gcn_resnet101(pretrained, dataset, t, adj_file=None, device='cpu', num_classes=80, in_channel=300):
    model = torch_models.resnet101(pretrained=pretrained)

    model = GCNResnet(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adj_file=adj_file)
    # 设置必要的信息
    set_model_input_shape_attr(model, dataset)
    model.dataset = dataset
    return model.to(device)
