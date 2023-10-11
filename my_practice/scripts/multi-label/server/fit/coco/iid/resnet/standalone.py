# 服务器与客户端的通用逻辑
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

import torch
import numpy as np
import torch.nn as nn
import torchvision.models as torch_models

# 导出transformer


# 全连接层中隐藏层的维度
# Todo: 即论文中"the dimension of feed-forward head"
d_ff = 512
# 向量K和Q的维度，V的维度可以不同，这里为了方便，令其维度相同
# Todo: 即论文中"the dimension of attention head"
d_k = d_v = 64
# 编码块的数量
n_layers = 3
# 多头，每套挖掘不同维度的注意力机制
n_heads = 4


class PositionalEncoding(nn.Module):
    # Todo: max_len是输入序列的长度，即H*W，和前面对接
    #  将位置编码设定为可学习的参数
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.encodings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        '''
        x: [batch_size, max_len, d_model]
        :param x:
        :return:
        '''
        # Todo: x是一个矩阵，直接加上对应的位置即可
        #  对self.encodings在batch_size维度上进行扩充
        x = x + self.encodings.weight.unsqueeze(0)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''

        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v, d_v]
        :return:
        '''
        # Todo: 这里为什么要除以d_k的平方根？不除应该也行
        # scores的维度为[batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # 返回加权值和注意力矩阵
        return context, attn


# Todo: 多头注意力的具体实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # Todo: 这一部分是在Decoder中实现的，Encoder中应该不用
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V):
        """
        Todo: 这里的len_q、len_k和len_v是什么意思？不一定相等吗？
         输入的嵌入向量在哪里？
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v, d_model]
        :return:
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # context的维度为：[batch_size, n_heads, len_q, d_v]
        context, _ = ScaledDotProductAttention()(Q, K, V)
        # Todo: 下面将不同头的输出向量拼接在一起
        #  拼接后context的维度为：[batch_size,len_q,n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # 进行投影，投影后的维度为[batch_size, len_q, d_model]
        output = self.fc(context)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # 前馈网络中也需要使用残差连接
        residual = inputs
        output = self.fc(inputs)
        # 输出的维度为[batch_size, seq_len, d_model ]
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # 自注意力模型，input_Q、input_K、input_V的输入都相等
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs的维度为[batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, d_model, seq_len, share):
        super(Encoder, self).__init__()
        # Todo: 这个其实就是之前的CNN学到的feature map
        # self.src_emb = nn.Embedding(seq_len, d_model)
        # 这里的位置嵌入是固定的，需要包含位置的相关信息，由于需要判别空间相关性，因此，该部分是必要的
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=seq_len)
        share_encoder_layer = EncoderLayer(d_model)

        self.layers = nn.ModuleList([share_encoder_layer if share else EncoderLayer(d_model) for _ in range(n_layers)])

    def forward(self, inputs):
        """
        inputs: [batch_size，seq_len，d_model]
        :param inputs:
        :return:
        """
        enc_outputs = self.pos_emb(inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        # enc_outputs的维度为[batch_size, src_len, embed_dim]
        return enc_outputs


class Transformer(nn.Module):
    def __init__(self, d_model, seq_len, share=False, device='cpu'):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model=d_model, seq_len=seq_len, share=share).to(device)

    def forward(self, inputs):
        self.encoder(inputs)


class Disentangler(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super(Disentangler, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_features, out_features=80):
        super(OutputLayer, self).__init__()
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.adaptive_avgpool(x)
        x = x.reshape(x.size(0), -1)  # 消除最后1
        x = self.fc(x)
        return x


class SSTModel(nn.Module):
    def __init__(self, channel_size=2048, height_width=14, num_labels=80):
        super(SSTModel, self).__init__()
        image_size = height_width * height_width
        model = torch_models.resnet101(pretrained=True, num_classes=1000)
        modules = list(model.children())[:-2]
        # 获取resnet101的features部分
        self.resnet = torch.nn.Sequential(*modules)
        # Todo: 空间加部分也共享参数
        self.spatial_transformer = Transformer(d_model=channel_size, seq_len=image_size, share=True)
        # Todo: 语义部分共享参数
        self.semantic_transformer = Transformer(d_model=image_size, seq_len=num_labels, share=True)
        # 将通道数量变为C，便于挖掘标签之间的语义相关性
        self.disentangler = Disentangler(channel_size, num_labels)
        # 第一个输出层，根据空间相关性特征计算输出
        self.output_layer1 = OutputLayer(channel_size)
        # 第二个输出层，根据语义相关性特征计算输出
        self.output_layer2 = OutputLayer(num_labels)
        self.channel_size = channel_size
        self.num_labels = num_labels
        # Todo: 这里是中间输出的feature map的宽度和高度
        self.height = height_width
        self.width = height_width

    # 输入[batch_size,channel=3,height,width = (448,448)]
    def forward(self, x):
        # 经过resnet后，[batch_size,2048,14,14]
        x = self.resnet(x)
        batch_size = x.size(0)
        # reshape到transformer可以接收的输入形式
        # reshape后x = [batch_size,2048,256]，输入到编码器中
        x = x.reshape(batch_size, self.channel_size, -1).transpose(1, 2)
        x = self.spatial_transformer.encoder(x)
        # 恢复到原来的维度
        x = x.transpose(1, 2).reshape(batch_size, self.channel_size, self.height, self.width)
        y1 = self.output_layer1(x)
        x = self.disentangler(x)
        x = x.reshape(batch_size, self.num_labels, -1)
        x = self.semantic_transformer.encoder(x)
        x = x.reshape(batch_size, self.num_labels, self.height, self.width)
        y2 = self.output_layer2(x)
        y = y1 + y2
        return y

import json
import os
import pickle


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


train_path = '/data/projects/iid_dataset/client1/train'
category_dir = '/data/projects/dataset'
# train_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
# category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

# 载入数据验证一下
train_dataset = COCO(train_path, config_dir=category_dir, transforms=transforms.Compose([
    # 将图像缩放为256*256
    transforms.Resize(512),
    # 随机裁剪出224*224大小的图像用于训练
    transforms.RandomResizedCrop(448),
    # 将图像进行水平翻转
    transforms.RandomHorizontalFlip(),
    # 转换为张量
    transforms.ToTensor(),
    # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

# Todo: 定义collate_fn
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=16, num_workers=16,
    drop_last=False, shuffle=False
)

device = 'cuda:3'
criterion = torch.nn.BCELoss()

sst_model = SSTModel().to(device)

# 配置需要优化的参数

optimizer = torch.optim.SGD(sst_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for epoch in range(100):
    for train_step, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        output = sst_model(inputs)
        # 进行sigmoid
        sigmoid_func = torch.nn.Sigmoid()

        loss = criterion(sigmoid_func(output), target)
        print(f'epoch = {epoch},loss= {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

