import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Parameter

from federatedml.nn.backend.gcn.models.graph_convolution import GraphConvolution
from federatedml.nn.backend.gcn.utils import gen_adjs


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


class SALGL(nn.Module):
    # 传进来一个resnet101，截取其前一部分作为主干网络
    def __init__(self, model, feat_dim=2048, att_dim=1024, num_scenes=6, in_channels=300,
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
        torch.nn.init.kaiming_normal_(self.scene_linear.weight)
        # 图卷积网络部分
        self.gc1 = GraphConvolution(in_channels, 1024)
        self.gc2 = GraphConvolution(1024, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        # Todo: register_buffer的好处
        self.register_buffer('comatrix', torch.zeros((num_scenes, num_classes, num_classes)))

        self.entropy = EntropyLoss()
        # 计算最大熵，预测向量呈均匀分布时的熵
        self.max_en = self.entropy(torch.tensor([1 / num_scenes] * num_scenes))

        # embed_dim是标签嵌入的维度，300维？
        # embed_dim = 300
        # self.attention = LowRankBilinearAttention(feat_dim, in_channels, att_dim)

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
        # 引入自连接
        for i in range(self.num_classes):
            comat[:, i, i] = 1
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
        img_feats = img_feats.view(img_feats.size(0), -1)  # [batch_size, feat_dim]

        scene_scores = self.scene_linear(img_feats)  # [batch_size, num_scenes]
        # 训练阶段更新comatrix
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
        batch_en = (self.max_en - self.entropy(_scene_probs)) * 1

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
        adjs = gen_adjs(comats)
        # 每张图片采用不同的邻接矩阵
        x = self.gc1(inp, adjs)
        x = self.relu(x)  # 进行leakyReLU
        x = self.gc2(x, adjs)  # [batch_size, num_classes, feat_dim]

        # 此时x的输出就是每张图像的分类器
        output = torch.matmul(x, img_feats.unsqueeze(2)).squeeze(-1)
        # Todo: 求点积，得到预测向量
        # 返回前向传播的结果
        return {
            'output': output,
            'scene_probs': scene_probs,
            'entropy_loss': sample_en + batch_en,
            'comat': comats
        }

    # 获取需要优化的变量
    def get_config_optim(self, lr=0.1, lrp=0.1):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.scene_linear.parameters(), 'lr': lr}  # 线性分类器应该不能聚合吧？
        ]
