from .components import ComponentMeta

distiller_cpn_meta = ComponentMeta("Distiller")


@distiller_cpn_meta.bind_param
def distiller_param():
    from federatedml.param.distiller_param import DistillerParam
    return DistillerParam


@distiller_cpn_meta.bind_runner.on_guest.on_host
def distiller_runer_client():
    from federatedml.nn.distiller.enter_point import DistillerDefaultClient
    return DistillerDefaultClient


@distiller_cpn_meta.bind_runner.on_arbiter
def distiller_runner_arbiter():
    from federatedml.nn.distiller.enter_point import DistillerDefaultServer
    return DistillerDefaultServer
