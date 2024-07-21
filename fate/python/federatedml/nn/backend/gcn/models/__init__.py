import torchvision.models as torch_models

from federatedml.nn.backend.gcn.models.full_salgl import FullSALGL
from federatedml.nn.backend.gcn.models.gin.ml_gin import GINResnet
from federatedml.nn.backend.gcn.models.instance_gcn.add_gcn import ADD_GCN
from federatedml.nn.backend.gcn.models.instance_gcn.dynamic_add_gcn import DYNAMIC_ADD_GCN
from federatedml.nn.backend.gcn.models.ml_gcn import GCNResnet, PGCNResnet
from federatedml.nn.backend.gcn.models.norm_gcn.norm_add_gcn import NORM_ADD_GCN
from federatedml.nn.backend.gcn.models.norm_gcn.norm_add_gin import NORM_ADD_GIN
from federatedml.nn.backend.gcn.models.papers.AAAI.add_gcn import AAAI_ADD_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.add_prob_gcn import AAAI_ADD_PROB_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.add_residual_gcn import AAAI_ADD_RESIDUAL_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.add_standard_gcn import AAAI_ADD_STANDARD_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.connect_add_gcn import AAAI_CONNECT_ADD_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.connect_add_prob_gcn import AAAI_CONNECT_ADD_PROB_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.connect_prob_residual_fix_static_gcn import \
    AAAI_CONNECT_PROB_RESIDUAL_FIX_STATIC_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.connect_prob_residual_gap_gcn import \
    AAAI_CONNECT_PROB_RESIDUAL_GAP_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.connect_prob_residual_gcn import AAAI_CONNECT_PROB_RESIDUAL_GCN
from federatedml.nn.backend.gcn.models.papers.AAAI.gin import AAAI_GIN
from federatedml.nn.backend.gcn.models.papers.AAAI.pruned_add_gcn import AAAI_PRUNED_ADD_GCN
from federatedml.nn.backend.gcn.models.papers.IJCNN.Agg_SALGL.resnet_agg_salgl import ResnetAggSalgl
from federatedml.nn.backend.gcn.models.papers.IJCNN.GCN.c_gcn import ResnetCGCN
from federatedml.nn.backend.gcn.models.papers.IJCNN.GCN.p_gcn import ResnetPGCN
from federatedml.nn.backend.gcn.models.papers.IJCNN.KMeans.resnet_kmeans import ResnetKmeans
from federatedml.nn.backend.gcn.models.papers.IJCNN.KMeans.vit_kmeans import VitKMeans
from federatedml.nn.backend.gcn.models.papers.IJCNN.SALGL.resnet_salgl import ResnetSalgl
from federatedml.nn.backend.gcn.models.pruned_add_gcn.connect_add_gcn import CONNECTED_ADD_GCN
from federatedml.nn.backend.gcn.models.pruned_add_gcn.connect_add_standard_gcn import CONNECT_ADD_STANDARD_GCN
from federatedml.nn.backend.gcn.models.pruned_add_gcn.pruned_add_gcn import PRUNED_ADD_GCN
from federatedml.nn.backend.gcn.models.pruned_add_gcn.pruned_add_gin import PRUNED_ADD_GIN
from federatedml.nn.backend.gcn.models.pruned_add_gcn.pruned_add_standard_gcn import PRUNED_ADD_STANDARD_GCN
from federatedml.nn.backend.gcn.models.salgl import SALGL
from federatedml.nn.backend.gcn.models.salgl_with_knn import SALGL_KNN


def gin_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=300,
                  out_channels=1024):
    model = torch_models.resnet101(pretrained)
    model = GINResnet(model=model, num_classes=num_classes, in_channels=in_channels,
                      out_channels=out_channels, adjList=adjList)
    return model.to(device)


def pruned_add_gcn_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                             out_channels=2048, needOptimize=True, constraint=False, prob=False, gap=False):
    model = torch_models.resnet101(pretrained)
    model = PRUNED_ADD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint, prob, gap)
    return model.to(device)


def pruned_add_standard_gcn_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                                      out_channels=2048, needOptimize=True, constraint=False, prob=False, gap=False):
    model = torch_models.resnet101(pretrained)
    model = PRUNED_ADD_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint,
                                    prob, gap)
    return model.to(device)


def pruned_add_gin_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                             out_channels=2048, needOptimize=True, constraint=False, prob=False, gap=False):
    model = torch_models.resnet101(pretrained)
    model = PRUNED_ADD_GIN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint,
                           prob, gap)
    return model.to(device)


def connect_add_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                             out_channels=2048, needOptimize=True, constraint=False, prob=False, gap=False):
    model = torch_models.resnet101(pretrained)
    model = CONNECT_ADD_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint,
                                     prob, gap)
    return model.to(device)


def connect_add_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                    out_channels=2048, needOptimize=True, constraint=False, prob=False, gap=False,
                    label_prob_vec=None):
    model = torch_models.resnet101(pretrained)
    model = CONNECTED_ADD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint,
                              prob, gap, label_prob_vec=label_prob_vec)
    return model.to(device)


# 使用add_gcn模型
def add_gcn_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                      out_channels=1024, needOptimize=True):
    model = torch_models.resnet101(pretrained)
    model = ADD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize)
    return model.to(device)


def dynamic_add_gcn_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                              out_channels=1024, needOptimize=True, constraint=False):
    model = torch_models.resnet101(pretrained)
    model = DYNAMIC_ADD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, constraint)
    return model.to(device)


# Todo: 对A进行约束 + 使用标准gcn
def norm_add_gcn_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                           out_channels=1024, needOptimize=True, norm_method='sigmoid'):
    model = torch_models.resnet101(pretrained)
    model = NORM_ADD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize, norm_method)
    return model.to(device)


# Todo: 对A进行约束 + 使用标准gin
def norm_add_gin_resnet101(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                           out_channels=1024, needOptimize=True):
    model = torch_models.resnet101(pretrained)
    model = NORM_ADD_GIN(model, num_classes, in_channels, out_channels, adjList, needOptimize)
    return model.to(device)


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


# Todo: 以下是IJCNN重写的模型

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


def resnet_c_gcn(pretrained, dataset, t, adjList=None, device='cpu', num_classes=80, in_channel=300):
    model = torch_models.resnet101(pretrained=pretrained)

    model = ResnetCGCN(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adjList=adjList)
    return model.to(device)


def resnet_p_gcn(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=2048, out_channel=1):
    model = torch_models.resnet101(pretrained=pretrained)
    model = ResnetPGCN(model=model, num_classes=num_classes, in_channel=in_channel, out_channels=out_channel,
                       adjList=adjList)
    return model.to(device)


def aaai_add_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                 out_channels=1024):
    model = torch_models.resnet101(pretrained)
    model = AAAI_ADD_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_add_residual_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                          out_channels=1024, needOptimize=True):
    model = torch_models.resnet101(pretrained)
    model = AAAI_ADD_RESIDUAL_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize)
    return model.to(device)


def aaai_add_standard_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                          out_channels=1024, prob=True, gap=False, needOptimize=True):
    model = torch_models.resnet101(pretrained)
    model = AAAI_ADD_STANDARD_GCN(model, num_classes, in_channels, out_channels, adjList, needOptimize)
    return model.to(device)


def aaai_add_prob_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                      out_channels=1024, prob=True, gap=False):
    model = torch_models.resnet101(pretrained)
    model = AAAI_ADD_PROB_GCN(model, num_classes, in_channels, out_channels, adjList,
                              prob=prob, gap=gap)
    return model.to(device)


def aaai_pruned_add_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                        out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_PRUNED_ADD_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_connect_add_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                         out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_CONNECT_ADD_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_connect_add_prob_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                              out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_CONNECT_ADD_PROB_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_connect_prob_residual_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                                   out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_CONNECT_PROB_RESIDUAL_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_connect_prob_residual_fix_static_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                                              out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_CONNECT_PROB_RESIDUAL_FIX_STATIC_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_connect_prob_residual_gap_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                                       out_channels=2048):
    model = torch_models.resnet101(pretrained)
    model = AAAI_CONNECT_PROB_RESIDUAL_GAP_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_fat_connect_add_gcn(pretrained, adjList, device='cpu', num_classes=80, in_channels=1024,
                             out_channels=2048):
    model = torch_models.resnet152(pretrained)
    model = AAAI_CONNECT_ADD_GCN(model, num_classes, in_channels, out_channels, adjList)
    return model.to(device)


def aaai_gin(pretrained, adjList, device='cpu', num_classes=80, in_channels=300,
             out_channels=1024):
    model = torch_models.resnet101(pretrained)
    model = AAAI_GIN(model=model, num_classes=num_classes, in_channels=in_channels,
                     out_channels=out_channels, adjList=adjList)
    return model.to(device)
