import torch.nn

from federatedml.nn.backend.distiller.models.imagenet.alexnet_batchnorm import AlexNetBN
from federatedml.nn.backend.distiller.utils import set_model_input_shape_attr
from federatedml.nn.backend.multi_label.models.lstm.cnn_rnn import CnnRnn
from federatedml.nn.backend.multi_label.models.srn.srn import *
from federatedml.nn.backend.multi_label.models.transformers.transformer import *

TORCHVISION_MODEL_NAMES = sorted(
    name for name in torch_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torch_models.__dict__[name]))

resnet_models = dict()
resnet_models[50] = torch_models.resnet50
resnet_models[101] = torch_models.resnet101


def create_model(pretrained, dataset, arch, device, num_classes=1000):
    model = None

    if dataset == 'imagenet' or dataset == 'ms-coco':
        # if num_classes == 10:
        # print("使用alexnet bn网络")
        model = AlexNetBN(num_classes=num_classes)
        # model = AlexNet(num_classes=num_classes)
        # else:
        #     model = _create_imagenet_model(arch, pretrained, num_classes=num_classes)
    set_model_input_shape_attr(model, dataset)
    model.arch = arch
    model.dataset = dataset
    return model.to(device)


def create_resnet101_model(pretrained, device, num_classes=80, layer_num=101):
    # Todo: 先下载1000类的全连接层
    model = resnet_models[layer_num](pretrained=pretrained, num_classes=1000)
    # 将最后的全连接层替换掉
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_classes))
    torch.nn.init.kaiming_normal_(model.fc[0].weight.data)
    return model.to(device)


def create_lstm_model(embed_size, hidden_size, num_layers, label_num, device):
    model = CnnRnn(embed_size, hidden_size, label_num, num_layers, device)
    return model


def _create_imagenet_model(arch, pretrained, num_classes):
    model = None
    if arch in TORCHVISION_MODEL_NAMES:
        model = getattr(torch_models, arch)(pretrained=pretrained, num_classes=num_classes)
    return model
