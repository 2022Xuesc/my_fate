import json

from federatedml.param.base_param import BaseParam


class GCNParam(BaseParam):
    def __init__(self,
                 aggregate_every_n_epoch: int = 1,
                 max_iter: int = 10,
                 batch_size: int = 32,
                 sched_dict: dict = None,
                 epochs: int = 500,
                 device: str = 'cpu',
                 pretrained: bool = True,
                 dataset: str = 'ms-coco',
                 early_stop_eps: float = 0.,
                 arch: str = 'gcn',
                 lr: float = 0.005,
                 num_labels: int = 80,
                 ):
        super(GCNParam, self).__init__()
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.pretrained = pretrained
        self.sched_dict = sched_dict
        self.dataset = dataset
        self.arch = arch
        self.lr = lr
        self.early_stop_eps = early_stop_eps
        self.num_labels = num_labels

    def check(self):
        pass

    # Todo: 编写proto文件
    def generate_pb(self):
        from federatedml.protobuf.generated import multi_label_meta_pb2
        pb = multi_label_meta_pb2.MultiLabelParam()
        pb.aggregate_every_n_epoch = self.aggregate_every_n_epoch
        pb.batch_size = self.batch_size
        pb.max_iter = self.max_iter
        pb.num_labels = self.num_labels
        for sched_info in self.sched_dict:
            pb.sched_dict.append(json.dumps(sched_info))

        pb.device = self.device
        pb.pretrained = self.pretrained
        pb.dataset = self.dataset
        pb.arch = self.arch
        pb.early_stop_eps = self.early_stop_eps
        pb.lr = self.lr
        return pb

    # Todo: protobuf
    def restore_from_pb(self, pb):
        self.aggregate_every_n_epoch = pb.aggregate_every_n_epoch
        self.max_iter = pb.max_iter
        self.batch_size = pb.batch_size
        for sched_info in pb.sched_dict:
            self.sched_dict.append(json.loads(sched_info))
        self.device = pb.device
        self.pretrained = pb.pretrained
        self.dataset = pb.dataset
        self.arch = pb.arch
        self.early_stop_eps = pb.early_stop_eps
        self.lr = pb.lr
        self.num_labels = pb.num_labels
        return pb
