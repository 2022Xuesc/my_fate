#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from .components import ComponentMeta

homo_lr_cpn_meta = ComponentMeta("HomoLR")


@homo_lr_cpn_meta.bind_param
def homo_lr_param():
    from federatedml.param.logistic_regression_param import HomoLogisticParam

    return HomoLogisticParam


@homo_lr_cpn_meta.bind_runner.on_guest
def homo_lr_runner_guest():
<<<<<<< HEAD
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_guest import (
        HomoLRGuest, )

    return HomoLRGuest
=======
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_client import (
        HomoLRClient, )

    return HomoLRClient
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e


@homo_lr_cpn_meta.bind_runner.on_host
def homo_lr_runner_host():
<<<<<<< HEAD
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_host import (
        HomoLRHost, )

    return HomoLRHost
=======
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_client import (
        HomoLRClient, )

    return HomoLRClient
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e


@homo_lr_cpn_meta.bind_runner.on_arbiter
def homo_lr_runner_arbiter():
<<<<<<< HEAD
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_arbiter import (
        HomoLRArbiter, )

    return HomoLRArbiter
=======
    from federatedml.linear_model.coordinated_linear_model.logistic_regression.homo_logistic_regression.homo_lr_server import (
        HomoLRServer, )

    return HomoLRServer
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
