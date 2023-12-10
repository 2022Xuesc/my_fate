import torchvision.models as torch_models

from federatedml.nn.backend.gcn.models.IJCNN.SALGL.resnet_salgl import ResnetSalgl
from federatedml.nn.backend.gcn.models.full_salgl import FullSALGL
from federatedml.nn.backend.gcn.models.ml_gcn import GCNResnet, PGCNResnet
from federatedml.nn.backend.gcn.models.salgl import SALGL
from federatedml.nn.backend.gcn.models.salgl_with_knn import SALGL_KNN
from federatedml.nn.backend.gcn.models.IJCNN.KMeans.resnet_kmeans import ResnetKmeans
from federatedml.nn.backend.gcn.models.IJCNN.KMeans.vit_kmeans import VitKMeans
from federatedml.nn.backend.gcn.models.IJCNN.Agg_SALGL.resnet_agg_salgl import ResnetAggSalgl


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


def full_salgl(pretrained, device, num_scenes=6, n_head=4, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return FullSALGL(model, num_scenes=num_scenes, n_head=n_head, num_classes=num_classes).to(device)


def salgl_knn(pretrained, device, num_scenes=6, n_head=4, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return SALGL_KNN(model, num_scenes=num_scenes, n_head=n_head, num_classes=num_classes).to(device)


def resnet_agg_salgl(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetAggSalgl(model, num_scenes=num_scenes, num_classes=num_classes).to(device)


def resnet_salgl(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetSalgl(model, num_scenes=num_scenes, num_classes=num_classes).to(device)


def resnet_kmeans(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetKmeans(model, num_scenes=num_scenes, num_classes=num_classes).to(device)


def vit_kmeans(pretrained, device, num_scenes=6, n_head=4, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return VitKMeans(model, num_scenes=num_scenes, num_classes=num_classes, n_head=n_head).to(device)
