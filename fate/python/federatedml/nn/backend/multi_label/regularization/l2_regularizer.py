from federatedml.nn.backend.multi_label.regularization._Regularizer import _Regularizer


class L2Regularizer(_Regularizer):
<<<<<<< HEAD
    def __init__(self, name,lamda):
        super(L2Regularizer, self).__init__(name,lamda)

    def loss(self, param, regularizer_loss):
        regularizer_loss += L2Regularizer.__add__l1(param)
        return regularizer_loss

    @staticmethod
    def __add__l1(var):
        return var.norm(2)
=======
    def __init__(self, name, lamda):
        super(L2Regularizer, self).__init__(name, lamda)

    def loss(self, param, regularizer_loss):
        regularizer_loss += param.norm(2) * self.lamda
        return regularizer_loss
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
