import torch.nn as nn
import torchvision.models as models

__all__ = ['EncoderCNN']


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