import torchvision.models as torch_models

from federatedml.nn.backend.distiller.utils import set_model_input_shape_attr
from federatedml.nn.backend.distiller.models.imagenet.alexnet_batchnorm import AlexNetBN


from federatedml.nn.backend.multi_label.models.lstm.cnn_rnn import CnnRnn

TORCHVISION_MODEL_NAMES = sorted(
    name for name in torch_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torch_models.__dict__[name]))


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


def create_lstm_model(embed_size, hidden_size, num_layers, label_num, device):
    model = CnnRnn(embed_size,hidden_size,label_num,num_layers,device)
    return model


def _create_imagenet_model(arch, pretrained, num_classes):
    model = None
    if arch in TORCHVISION_MODEL_NAMES:
        model = getattr(torch_models, arch)(pretrained=pretrained, num_classes=num_classes)
    return model
