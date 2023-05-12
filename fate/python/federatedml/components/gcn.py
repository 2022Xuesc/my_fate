from .components import ComponentMeta

gcn_cpn_meta = ComponentMeta("GCN")


@gcn_cpn_meta.bind_param
def multi_label_param():
    from federatedml.param.gcn_param import GCNParam
    return GCNParam


# Todo: 定义客户端和服务器端的行为
@gcn_cpn_meta.bind_runner.on_guest.on_host
def gcn_runer_client():
    from federatedml.nn.gcn.enter_point import GCNDefaultClient
    return GCNDefaultClient


@gcn_cpn_meta.bind_runner.on_arbiter
def gcn_runner_arbiter():
    from federatedml.nn.gcn.enter_point import GCNDefaultServer
    return GCNDefaultServer
