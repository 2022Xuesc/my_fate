__all__ = ['Transformer', 'Disentangler', 'OutputLayer', 'SSTModel']

import torch
import numpy as np
import torch.nn as nn
import torchvision.models as torch_models

# 导出transformer


# 全连接层中隐藏层的维度
d_ff = 2048
# 向量K和Q的维度，V的维度可以不同，这里为了方便，令其维度相同
d_k = d_v = 128
# encoder block和decoder block的个数
n_layers = 2
# 多头，每套挖掘不同维度的注意力机制
n_heads = 8


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
    def __init__(self, d_model, seq_len=49, share=False, device='cpu'):
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
    def __init__(self, channel_size=2048, num_labels=80):
        super(SSTModel, self).__init__()
        model = torch_models.resnet101(pretrained=True, num_classes=1000)
        modules = list(model.children())[:-2]
        # 获取resnet101的features部分
        self.resnet = torch.nn.Sequential(*modules)

        self.spatial_transformer = Transformer(d_model=channel_size)
        # Todo: 语义部分共享参数
        self.semantic_transformer = Transformer(d_model=80, share=True)
        # 将通道数量变为C，便于挖掘标签之间的语义相关性
        self.disentangler = Disentangler(channel_size, num_labels)
        # 第一个输出层，根据空间相关性特征计算输出
        self.output_layer1 = OutputLayer(channel_size)
        # 第二个输出层，根据语义相关性特征计算输出
        self.output_layer2 = OutputLayer(num_labels)
        self.channel_size = channel_size
        self.num_labels = num_labels
        # Todo: 这里是中间输出的feature map的宽度和高度
        self.height = 7
        self.width = 7

    def forward(self, x):
        x = self.resnet(x)
        batch_size = x.size(0)
        # reshape到transformer可以接收的输入形式
        x = x.reshape(batch_size, self.channel_size, -1).transpose(1, 2)
        x = self.spatial_transformer.encoder(x)
        # 恢复到原来的维度
        x = x.transpose(1, 2).reshape(batch_size, self.channel_size, self.height, self.width)
        y1 = self.output_layer1(x)
        x = self.disentangler(x)
        x = x.reshape(batch_size, self.num_labels, -1).transpose(1, 2)
        x = self.semantic_transformer.encoder(x)
        x = x.transpose(1, 2).reshape(batch_size, self.num_labels, self.height, self.width)
        y2 = self.output_layer2(x)
        y = y1 + y2
        return y
