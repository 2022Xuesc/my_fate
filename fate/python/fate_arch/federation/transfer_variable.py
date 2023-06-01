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

import hashlib
import typing
from typing import Union

from fate_arch.common import Party, profile
from fate_arch.common.log import getLogger
from fate_arch.federation._gc import IterationGC

__all__ = ["Variable", "BaseTransferVariables"]

LOGGER = getLogger()


class FederationTagNamespace(object):
    __namespace = "default"

    @classmethod
    def set_namespace(cls, namespace):
        cls.__namespace = namespace

    @classmethod
    def generate_tag(cls, *suffix):
        tags = (cls.__namespace, *map(str, suffix))
        return ".".join(tags)


class Variable(object):
    """
    variable to distinguish federation by name
    """

    __instances: typing.MutableMapping[str, "Variable"] = {}

    @classmethod
    def get_or_create(
        cls, name, create_func: typing.Callable[[], "Variable"]
    ) -> "Variable":
        if name not in cls.__instances:
            value = create_func()
            cls.__instances[name] = value
        return cls.__instances[name]

    def __init__(
        self, name: str, src: typing.Tuple[str, ...], dst: typing.Tuple[str, ...]
    ):

        if name in self.__instances:
            raise RuntimeError(
                f"{self.__instances[name]} with {name} already initialized, which expected to be an singleton object."
            )

        assert (
            len(name.split(".")) >= 3
        ), "incorrect name format, should be `module_name.class_name.variable_name`"
        # 指明名称、源和目的地址
        self._name = name
        self._src = src
        self._dst = dst
        self._get_gc = IterationGC()
        self._remote_gc = IterationGC()
        self._use_short_name = True
        self._short_name = self._get_short_name(self._name)

    @staticmethod
    def _get_short_name(name):
        fix_sized = hashlib.blake2b(name.encode("utf-8"), digest_size=10).hexdigest()
        _, right = name.rsplit(".", 1)
        return f"hash.{fix_sized}.{right}"

    # copy never create a new instance
    def __copy__(self):
        return self

    # deepcopy never create a new instance
    def __deepcopy__(self, memo):
        return self

    def set_preserve_num(self, n):
        self._get_gc.set_capacity(n)
        self._remote_gc.set_capacity(n)
        return self

    def disable_auto_clean(self):
        self._get_gc.disable()
        self._remote_gc.disable()
        return self

    def clean(self):
        self._get_gc.clean()
        self._remote_gc.clean()

    def remote_parties(
        self,
        obj,
        parties: Union[typing.List[Party], Party],
        suffix: Union[typing.Any, typing.Tuple] = tuple(),
    ):
        """
        将对象发送给指定的参与方

        Parameters
        ----------
        obj: object or table
           object or table to remote
        parties: typing.List[Party]
           parties to remote object/table to
        suffix: str or tuple of str
           suffix used to distinguish federation with in variable

        Returns
        -------
        None
        """
        from fate_arch.session import get_session
        # 获取会话
        session = get_session()
        if isinstance(parties, Party):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)# 这里将聚合轮次附着到了tag上

        # 验证是否能将对象发送给parties
        for party in parties:
            if party.role not in self._dst:
                raise RuntimeError(
                    f"not allowed to remote object to {party} using {self._name}"
                )
        # 验证源是否符合要求
        local = session.parties.local_party.role
        if local not in self._src:
            raise RuntimeError(
                f"not allowed to remote object from {local} using {self._name}"
            )

        name = self._short_name if self._use_short_name else self._name

        timer = profile.federation_remote_timer(name, self._name, tag, local, parties)

        # Todo: 将对象发送给parties，这里是实际的发送过程
        session.federation.remote(
            v=obj, name=name, tag=tag, parties=parties, gc=self._remote_gc
        )
        timer.done(session.federation)

        self._remote_gc.gc()

    def get_parties(
        self,
        parties: Union[typing.List[Party], Party],
        sync: bool = True,
        suffix: Union[typing.Any, typing.Tuple] = tuple(),
    ):
        """
        Todo: 从指定的parties获取对象或表

        Parameters
        ----------
        parties: typing.List[Party]
           parties to remote object/table to
        suffix: str or tuple of str
           suffix used to distinguish federation with in variable
        sync: bool
        Returns
        -------
        list
           a list of objects/tables get from parties with same order of ``parties``

        """
        from fate_arch.session import get_session

        session = get_session()
        if not isinstance(parties, list):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)# 这里tag有问题，因为每个客户端的聚合轮次不同

        for party in parties:
            if party.role not in self._src:
                raise RuntimeError(
                    f"not allowed to get object from {party} using {self._name}"
                )
        local = session.parties.local_party.role
        if local not in self._dst:
            raise RuntimeError(
                f"not allowed to get object to {local} using {self._name}"
            )

        name = self._short_name if self._use_short_name else self._name
        timer = profile.federation_get_timer(name, self._name, tag, local, parties)
        # 从federation中获取对象
        # Todo: 具体的方法调用的是？
        rtn = session.federation.get(
            name=name, tag=tag, parties=parties, gc=self._get_gc,sync=sync
        )
        timer.done(session.federation)

        self._get_gc.gc()

        return rtn

    def remote(self, obj, role=None, idx=-1, suffix=tuple()):
        """
        将对象obj发送给其他参与方

        Args:
            obj: object to be sent
            role: role of parties to sent to, use one of ['Host', 'Guest', 'Arbiter', None].
                The default is None, means sent values to parties regardless their party role
            idx: id of party to sent to.
                The default is -1, which means sent values to parties regardless their party id
            suffix: additional tag suffix, the default is tuple()
        """
        from fate_arch.session import get_parties

        party_info = get_parties()
        if idx >= 0 and role is None:
            raise ValueError("role cannot be None if idx specified")

        # 在运行时配置中获取目的角色的子集
        if role is None:
            parties = party_info.roles_to_parties(self._dst, strict=False)
        else:
            if isinstance(role, str):
                role = [role]
            parties = party_info.roles_to_parties(role)

        if idx >= 0:
            if idx >= len(parties):
                raise RuntimeError(
                    f"try to remote to {idx}th party while only {len(parties)} configurated: {parties}, check {self._name}"
                )
            parties = parties[idx]
        # 调用本地方法进行发送
        return self.remote_parties(obj=obj, parties=parties, suffix=suffix)

    # 实际的获取方法
    def get(self, idx=-1, role=None, suffix=tuple()):
        """
        从其他参与方接收对象obj

        Args:
            idx: id of party to get from.
                The default is -1, which means get values from parties regardless their party id
            suffix: additional tag suffix, the default is tuple()

        Returns:
            object or list of object
        """
        from fate_arch.session import get_parties

        if role is None:
            src_parties = get_parties().roles_to_parties(roles=self._src, strict=False)
        else:
            if isinstance(role, str):
                role = [role]
            src_parties = get_parties().roles_to_parties(roles=role, strict=False)
        if isinstance(idx, list):
            rtn = self.get_parties(parties=[src_parties[i] for i in idx], suffix=suffix)
        elif isinstance(idx, int):
            if idx < 0:
                rtn = self.get_parties(parties=src_parties, suffix=suffix)
            else:
                if idx >= len(src_parties):
                    raise RuntimeError(
                        f"try to get from {idx}th party while only {len(src_parties)} configurated: {src_parties}, check {self._name}"
                    )
                rtn = self.get_parties(parties=src_parties[idx], suffix=suffix)[0]
        else:
            raise ValueError(
                f"illegal idx type: {type(idx)}, supported types: int or list of int"
            )
        return rtn


# Todo: 基本传输变量
class BaseTransferVariables(object):
    def __init__(self, *args):
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # Todo: 设置流id，这里的流id指的是？
    @staticmethod
    def set_flowid(flowid):
        """
        为联邦设置全局的命名空间
        Parameters
        ----------
        flowid: str
           namespace

        Returns
        -------
        None

        """
        FederationTagNamespace.set_namespace(str(flowid))

    # 创建变量
    # 这里的src是源地址，dst是目的地址
    def _create_variable(
        self, name: str, src: typing.Iterable[str], dst: typing.Iterable[str]
    ) -> Variable:
        # 获取全名称
        full_name = f"{self.__module__}.{self.__class__.__name__}.{name}"
        return Variable.get_or_create(
            full_name, lambda: Variable(name=full_name, src=tuple(src), dst=tuple(dst))
        )

    @staticmethod
    def all_parties():
        """
        获取所有的参与方

        Returns
        -------
        list
           参与方列表

        """
        from fate_arch.session import get_parties
        return get_parties().all_parties

    @staticmethod
    def local_party():
        """
        获取本地参与方
        Todo: 本地参与方指的是当前参与方吗？

        Returns
        -------
        Party
           party this program running on

        """
        from fate_arch.session import get_parties

        return get_parties().local_party
