import numpy
import torch
import torch.nn as nn
import torchvision.models as torch_models


class MainOutputLayer(nn.Module):
    def __init__(self, modules):
        super(MainOutputLayer, self).__init__()
        self.layer4 = modules[0]
        self.avg_pool = modules[1]
        self.fc = nn.Linear(in_features=2048, out_features=80)
        torch.nn.init.kaiming_normal_(self.fc.weight.data)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, num_labels):
        super(AttentionLayer, self).__init__()
        # Todo: 手推一下resnet卷积层的变换规则，设置好一下三个卷积层的参数
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1))
        # Todo: 这里不进行下采样，因此不设置步长，又因为kernel_size=3，将padding设置为1
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=num_labels, kernel_size=(1, 1))

    # x的输入是[batch_size,channel_size,height,width]
    #        [batch_size,1024,14,14]
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y_data = x.data.cpu().numpy()
        numpy.save('/home/klaus125/research/fate/my_practice/knowledge/stats/attention_maps', y_data)

        # 此时x的维度是[batch_size,C,14,14]
        # 按照3，4维度进行
        exp_x = torch.exp(x)
        sum_exp_x = exp_x.sum(dim=[2, 3], keepdim=True)
        y = exp_x / sum_exp_x
        # 将y收集起来
        return y


class ConfidenceLayer(nn.Module):
    def __init__(self, num_labels):
        super(ConfidenceLayer, self).__init__()
        # 使用1*1卷积层实现全连接层的功能
        self.conv = nn.Conv2d(in_channels=1024, out_channels=num_labels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x


# 定义分组二维卷积层数
class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=512):
        super(GroupConv2d, self).__init__()
        in_channel_per_group = in_channels // groups
        out_channel_per_group = out_channels // groups
        self.conv_list = nn.ModuleList()
        for i in range(groups):
            self.conv_list.append(nn.Conv2d(
                in_channels=in_channel_per_group,
                out_channels=out_channel_per_group,
                kernel_size=kernel_size,
                bias=False
            ))

    def forward(self, x):
        # 将输入张量按通道数分成多个分组
        xs = torch.split(x, split_size_or_sections=int(x.size(1) / len(self.conv_list)), dim=1)
        # 对分一个分组应用对应的卷积层并且拼接结果
        conv_outputs = [conv(x_) for conv, x_ in zip(self.conv_list, xs)]
        return torch.cat(conv_outputs, dim=1)


class SpatialRegularizationLayer(nn.Module):
    def __init__(self, num_labels):
        super(SpatialRegularizationLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_labels, out_channels=512, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1))
        self.conv3 = GroupConv2d(in_channels=512, out_channels=2048, kernel_size=(14, 14))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=2048, out_features=num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        return y


class SRN(nn.Module):
    def __init__(self, num_labels=80):
        super(SRN, self).__init__()
        # 获取预训练好的resnet101，去除掉feature map之后的层
        # Todo: 这里还要剩下3个输出通道为2048的构建块
        modules = list(torch_models.resnet101(pretrained=True, num_classes=1000).children())
        self.resnet = nn.Sequential(*modules[:-3])
        # 定义主分支之后的分类和预测层
        self.main_output_layer = MainOutputLayer(modules[-3:])
        # 定义得到注意力映射的模块（3个卷积层：1*1、3*3、1*1）
        # 定义得到置信度映射的模块（1*1卷积层）
        # 定义得到注意力损失的预测模块
        # 定义得到空间正则化的预测模块
        self.att_layer = AttentionLayer(num_labels=num_labels)
        self.confidence_layer = ConfidenceLayer(num_labels=num_labels)
        self.spatial_regularization_layer = SpatialRegularizationLayer(num_labels=num_labels)

    def forward(self, x):
        features = self.resnet(x)
        y_cls = self.main_output_layer(features)
        # 计算注意力映射和置信度映射
        attn = self.att_layer(features)
        confidence = self.confidence_layer(features)
        product = torch.mul(attn, confidence)
        # 对其2，3维度进行求和，且无需保持维度
        y_att = product.sum(dim=[2, 3], keepdim=False)

        # 计算y_sr
        # 对product进行sigmoid，直接调用函数即可，对于每个数都求sigmoid，将其变为[0,1]
        normalized_attn = torch.sigmoid(product)
        y_sr = self.spatial_regularization_layer(normalized_attn)
        # 将y_cls和y_sr相加得到y_hat
        y_hat = y_cls + y_sr
        # Todo: 我采用联合优化的方式，返回y_att的y_att和，sigmoid后和标签求BCE Loss
        return y_att + y_hat
