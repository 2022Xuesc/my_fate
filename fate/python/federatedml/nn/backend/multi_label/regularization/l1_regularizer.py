from federatedml.nn.backend.multi_label.regularization._Regularizer import _Regularizer


class L1Regularizer(_Regularizer):
    def __init__(self, name,lamda):
        super(L1Regularizer, self).__init__(name,lamda)

    def loss(self, param, regularizer_loss):
        regularizer_loss += L1Regularizer.__add__l1(param) * self.lamda
        return regularizer_loss

    @staticmethod
    def __add__l1(var):
        return var.norm(1)
