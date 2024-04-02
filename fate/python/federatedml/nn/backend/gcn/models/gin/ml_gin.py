from torch.nn import Parameter

from federatedml.nn.backend.gcn.models.gin.gin_layer import GINLayer
from federatedml.nn.backend.gcn.utils import *


class GINResnet(nn.Module):
    def __init__(self, model, num_classes, in_channels=300, out_channels=2048,
                 latent_dim=1024, adjList=None):
        super(GINResnet, self).__init__()
        self.A = adjList
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
        # Todo: 这里使用MaxPool2d好还是AdaptivePool好？之后进行验证
        self.pooling = nn.MaxPool2d(14, 14)

        # 定义图同构层
        self.gin1 = GINLayer(in_channels, 1024)
        self.gin2 = GINLayer(1024, out_channels)

        # 定义重要性向量层
        self.imp_layer = nn.Linear(out_channels, latent_dim, True)
        self.latent_layer = nn.Linear(2048, latent_dim, True)
        self.map_layer = nn.Linear(latent_dim, latent_dim, True)

        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # 最后全连接分类器层
        self.classifier = nn.Linear(latent_dim, 1, True)

    def generateA(self, adjList):
        return Parameter(torch.from_numpy(adjList).float())

    def updateA(self, adjList):
        self.A.data.copy_(torch.from_numpy(adjList).float())

    # 前向传播逻辑：features --> pooling -->
    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        inp = inp[0]  # 从数据集中获取初始标签嵌入
        x = self.gin1(inp, self.A)
        x = self.gin2(x, self.A)

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
