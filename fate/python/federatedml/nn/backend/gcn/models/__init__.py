import torchvision.models as torch_models

from federatedml.nn.backend.gcn.models.full_salgl import FullSALGL
from federatedml.nn.backend.gcn.models.ml_gcn import GCNResnet, PGCNResnet
from federatedml.nn.backend.gcn.models.salgl import SALGL


def gcn_resnet101(pretrained, dataset, t, adjList=None, device='cpu', num_classes=80, in_channel=300):
    model = torch_models.resnet101(pretrained=pretrained)

    model = GCNResnet(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adjList=adjList)
    return model.to(device)


def p_gcn_resnet101(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=2048, out_channel=1):
    model = torch_models.resnet101(pretrained=pretrained)
    model = PGCNResnet(model=model, num_classes=num_classes, in_channel=in_channel, out_channels=out_channel,
                       adjList=adjList)
    return model.to(device)


def sal_gl(pretrained, device):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return SALGL(model).to(device)


def full_salgl(pretrained, device):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return FullSALGL(model).to(device)
