from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.model_base import ModelBase, MetricMeta, Metric
from federatedml.param.distiller_param import DistillerParam
from federatedml.util import consts


class DistillerBase(ModelBase):

    def __init__(self, trans_var):
        super().__init__()
        self.model_param = DistillerParam()
        self.transfer_variable = trans_var

    def _init_model(self, param):
        self.param = param


class DistillerServer(DistillerBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self.aggregator = ...
        self._init_iteration = 0

    def _init_model(self, param: DistillerParam):
        super()._init_model(param)

    def callback_loss(self, iter_num, loss):
        # noinspection PyTypeChecker
        metric_meta = MetricMeta(
            name="train",
            metric_type="LOSS",
            extra_metas={
                "unit_name": "iters",
            },
        )

        self.callback_meta(
            metric_name="loss", metric_namespace="train", metric_meta=metric_meta
        )
        self.callback_metric(
            metric_name="loss",
            metric_namespace="train",
            metric_data=[Metric(iter_num, loss)],
        )

    def fit(self, train_data, valid_data):
        self.callback_list.on_train_begin({train_data, valid_data}, None)
        from federatedml.nn.distiller._fate import build_aggregator
        self.aggregator = build_aggregator(
            self.param, init_iteration=self._init_iteration
        )
        if not self.component_properties.is_warm_start:
            self.aggregator.dataset_align()
        self.aggregator.fit(self.callback_loss)
        self.callback_list.on_train_end()


class DistillerClient(DistillerBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self._compressor = ...

    def _init_model(self, param: DistillerParam):
        super()._init_model(param)

    def fit(self, train_data, valid_data):
        self.callback_list.on_train_begin({train_data, valid_data}, None)
        from federatedml.nn.distiller._fate import build_compressor
        self._compressor = None
        self._compressor, train_loader, valid_loader = build_compressor(
            param=self.param,
            train_data=train_data,
            valid_data=valid_data,
            should_label_align=not self.component_properties.is_warm_start,
            compressor=self._compressor
        )
        self._compressor.fit(train_loader, valid_loader)
        self.set_summary(self._compressor.summary())
        self.callback_list.on_train_end()


class DistillerDefaultTransVar(HomoTransferBase):
    def __init__(
            self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None
    ):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.secure_aggregator_trans_var = SecureAggregatorTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.loss_scatter_trans_var = LossScatterTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.has_converged_trans_var = HasConvergedTransVar(
            server=server, clients=clients, prefix=self.prefix
        )


class DistillerDefaultClient(DistillerClient):
    def __init__(self):
        super().__init__(trans_var=DistillerDefaultTransVar())


class DistillerDefaultServer(DistillerServer):
    def __init__(self):
        super().__init__(trans_var=DistillerDefaultTransVar())
