import torch.nn as nn
import torch
from torch.nn.utils.rnn import *

__all__ = ['LabelRNN']


class LabelRNN(nn.Module):
    # Todo: 这里max_seq_length指的是标签的生成长度吗？
    def __init__(self, embed_size, hidden_size, label_num, num_layers, max_seq_length):
        super(LabelRNN, self).__init__()
        self.embed = nn.Embedding(label_num, embed_size)
        # 设置batch_first为True，输入和输出的batch会在第一维，hn和cn不变
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, label_num)
        self.max_seq_length = max_seq_length

    # lengths表示输入labels的真实长度
    def forward(self, features, labels, lengths):
        embeddings = self.embed(labels)
        # 特征作为第0时刻的嵌入输入
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    # 给定图像特征，使用贪婪搜索生成图像描述
    # Todo: 预测阶段的工作
    def sample(self, features, states=None):
        sample_ids = []
        # 给inputs的第一维扩充，以输入到lstm中
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            # hiddens: (batch_size,1,hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # 将第一维消除掉
            # outputs: (batch_size,label_num)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sample_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids
