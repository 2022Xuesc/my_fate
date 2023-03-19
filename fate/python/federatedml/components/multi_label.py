from .components import ComponentMeta

multi_label_cpn_meta = ComponentMeta("MultiLabel")


@multi_label_cpn_meta.bind_param
def multi_label_param():
    from federatedml.param.multi_label_param import MultiLabelParam
    return MultiLabelParam


# Todo: 定义客户端和服务器端的行为
@multi_label_cpn_meta.bind_runner.on_guest.on_host
def multi_label_runer_client():
    from federatedml.nn.multi_label.enter_point import MultiLabelDefaultClient
    return MultiLabelDefaultClient


@multi_label_cpn_meta.bind_runner.on_arbiter
def multi_label_runner_arbiter():
    from federatedml.nn.multi_label.enter_point import MultiLabelDefaultServer
    return MultiLabelDefaultServer
