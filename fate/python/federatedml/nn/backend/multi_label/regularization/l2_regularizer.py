from federatedml.nn.backend.multi_label.regularization._Regularizer import _Regularizer


class L2Regularizer(_Regularizer):
    def __init__(self, name, lamda):
        super(L2Regularizer, self).__init__(name, lamda)

    def loss(self, param, regularizer_loss):
        regularizer_loss += param.norm(2) * self.lamda
        return regularizer_loss
