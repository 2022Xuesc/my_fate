import os.path

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import *
from torchvision import transforms
from data_loader import get_loader
import numpy as np
import csv
from PIL import Image

buf_size = 1
train_file = open('train.csv', 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(['epoch', 'loss', 'precision', 'recall'])

val_file = open('valid.csv', 'w', buffering=buf_size)
val_writer = csv.writer(val_file)
val_writer.writerow(['epoch', 'loss', 'precision', 'recall'])

# 定义数据的处理方式
data_transforms = {
    'train': transforms.Compose([
        # 将图像缩放为256*256
        transforms.Resize(256),
        # 随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(227),
        # 将图像进行水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试样本进行中心裁剪，且无需翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def load_image(image_path):
    # 以RGB格式打开图像
    # Pytorch DataLoader使用PIL的Image格式
    # 需要读取灰度图像时使用 .convert('')
    return Image.open(image_path).convert('RGB')


# Todo: 添加batchnorm层
class EncoderCNN(nn.Module):
    def __init__(self, label_num):
        super(EncoderCNN, self).__init__()
        # 获取原模型
        alexnet = models.alexnet(pretrained=False)

        # 获取最后一层的输入
        num_input = alexnet.classifier[6].in_features

        feature_model = list(alexnet.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_input, label_num))
        # 重构分类器
        alexnet.classifier = nn.Sequential(*feature_model)

        # 使用修改后的cnn模型
        self.cnn = alexnet

    def forward(self, images):
        return self.cnn(images)


# 输出是维度为label_num的向量
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


# 开始训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = '/home/klaus125/research/dataset/guest/train'
val_dir = '/home/klaus125/research/dataset/guest/val'
train_label_path = os.path.join(train_dir, 'embedding_labels.txt')
val_label_path = os.path.join(val_dir, 'embedding_labels.txt')

model_path = "checkpoints"

batch_size = 32
shuffle = True
num_workers = 8

train_loader = get_loader(label_path=train_label_path,
                          images_dir=train_dir,
                          transform=data_transforms['train'],
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers)
valid_loader = get_loader(label_path=val_label_path,
                          images_dir=val_dir,
                          transform=data_transforms['val'],
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers)
# 保留
test_loader = ...



# 这里可以直接使用sample方法
def predict(encoder, decoder, image, idx2label):
    with torch.no_grad():
        image_tensor = image.to(device)

        # 从图像中生成caption
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # 将label_ids转换为labels
        # Todo: 如果中间有end的label_id，则预测结束
        sampled_labels = []
        for label_id in sampled_ids:
            label = idx2label[label_id]
            sampled_labels.append(label)
            if label == '<end>':
                break


def train_one_epoch(encoder, decoder, criterion, optimizer, epoch, num_epochs, ):
    total_step = len(train_loader)
    for i, (images, labels, lengths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        packed = pack_padded_sequence(labels, lengths, batch_first=True)
        targets = packed[0]

        # 前向传播
        features = encoder(images)
        outputs = decoder(features, labels, lengths)
        # Todo: 根据输出和目标计算precision和recall
        #  targets的维度是164*1，表示164个位置中，正确的标签值
        #  outputs的维度是164*92，表示164个位置中，每个标签的预测概率，选择max即可

        # Todo: 这里可能会取出重复标签，但训练时候先不管，因为重复表明与target不符，
        #  会被优化器主动优化
        predicted_data = torch.argmax(outputs, dim=1)
        predict_labels = pad_packed_sequence(packed._replace(data=predicted_data), batch_first=True)
        # 对比labels计算平均precision和recall
        accuracy = calculate_accuracy(labels, predict_labels[0], lengths)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},accuracy: {:.4f}'
                  .format(epoch, num_epochs, i, total_step, loss.item(), accuracy))


def valid_one_epoch(encoder, decoder, epoch, num_epochs):
    total_step = len(valid_loader)
    with torch.no_grad():
        for i, (images, labels, lengths) in enumerate(valid_loader):
            images = images.to(device)
            features = encoder(images)
            # Todo: 注意，这里labels并不会作为输入，而是作为输出的评价
            #  如何批量进行验证呢？
            sample_ids = decoder.sample(features)
            precision, recall = calculate_precision_and_recall(labels, sample_ids, lengths)
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], precision: {:.4f},recall: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, precision, recall))


def calculate_precision_and_recall(labels, sample_ids, lengths):
    total_labels = sum(lengths)
    total_predict = 0
    right_predict = 0
    for end, label, sample_id in zip(lengths, labels, sample_ids):
        # 真实的标签
        label_set = set(label[:end].cpu().tolist())
        # 预测的标签
        indicators = torch.nonzero(sample_id == 91)
        predict_end = len(sample_id) if len(indicators) == 0 else indicators[0].item() + 1
        predict_set = set(sample_id[:predict_end].cpu().tolist())

        right_predict += len(predict_set.intersection(label_set))
        total_predict += len(predict_set)
    return right_predict / total_predict, right_predict / total_labels


# Todo: 训练时，有训练样本的标签时序输入，精度作为参考
def calculate_accuracy(labels, predict_labels, lengths):
    total = sum(lengths)
    right = 0
    for end, label, predict_label in zip(lengths, labels, predict_labels):
        label_set = set(label[:end].cpu().tolist())
        predict_label_set = set(predict_label[:end].cpu().tolist())
        right += len(label_set.intersection(predict_label_set))
    return right / total


# 训练时的一些参数
embed_size = 8
hidden_size = 4
num_layers = 1
label_num = 92
learning_rate = 0.005
num_epochs = 500
log_step = 5
save_step = 20

# Build the models
encoder = EncoderCNN(embed_size).to(device)
# Todo: 最多预测5个标签
decoder = LabelRNN(embed_size, hidden_size, label_num, num_layers, max_seq_length=5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.cnn.parameters())

optimizer = torch.optim.Adam(params, lr=learning_rate)


def train_and_validate(encoder, decoder, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_one_epoch(encoder, decoder, criterion, optimizer, epoch, num_epochs)
        valid_one_epoch(encoder, decoder, epoch, num_epochs)


train_and_validate(encoder, decoder, criterion, optimizer, num_epochs)
