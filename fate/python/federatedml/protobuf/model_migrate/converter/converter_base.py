#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


from abc import ABC, abstractmethod
from typing import Dict, Tuple
<<<<<<< HEAD
from federatedml.util.anonymous_generator import generate_anonymous
=======
from federatedml.util.anonymous_generator_util import Anonymous
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
from federatedml.util import consts


class AutoReplace(object):

    def __init__(self, guest_mapping, host_mapping, arbiter_mapping):
        self._mapping = {
            consts.GUEST: guest_mapping,
            consts.HOST: host_mapping,
            consts.ARBITER: arbiter_mapping
        }
<<<<<<< HEAD
=======
        self._anonymous_generator = Anonymous(migrate_mapping=self._mapping)
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e

    def get_mapping(self, role: str):
        if role not in self._mapping:
            raise ValueError('this role contains no site name {}'.format(role))
        return self._mapping[role]

<<<<<<< HEAD
    def anonymous_format(self, string: str):
        """{role}_{party_id}_{idx}"""
        role, party_id, idx = string.split('_')
        mapping = self.get_mapping(role)
        new_party_id = mapping[int(party_id)]
        return generate_anonymous(idx, new_party_id, role)

=======
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
    def party_tuple_format(self, string: str):
        """({role},{party_id})"""
        role, party_id = string.strip("()").split(",")
        return f"({role}, {self._mapping[role][int(party_id)]})"

    def colon_format(self, string: str):
        """{role}:{party_id}"""
        role, party_id = string.split(':')
        mapping = self.get_mapping(role)
        new_party_id = mapping[int(party_id)]
        return role + ':' + str(new_party_id)

    def maybe_anonymous_format(self, string: str):
<<<<<<< HEAD
        try:
            return self.anonymous_format(string)
        except Exception:
=======
        if self._anonymous_generator.is_anonymous(string):
            return self.migrate_anonymous_header([string])[0]
        else:
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
            return string

    def plain_replace(self, old_party_id, role):
        old_party_id = int(old_party_id)
        mapping = self._mapping[role]
        if old_party_id in mapping:
            return str(mapping[int(old_party_id)])
        return str(old_party_id)

<<<<<<< HEAD
=======
    def migrate_anonymous_header(self, anonymous_header):
        if isinstance(anonymous_header, list):
            return self._anonymous_generator.migrate_anonymous(anonymous_header)
        else:
            return self._anonymous_generator.migrate_anonymous([anonymous_header])[0]

>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
    def replace(self, string):

        if ':' in string:
            return self.colon_format(string)
<<<<<<< HEAD
        elif '_' in string:
            return self.anonymous_format(string)
=======
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
        else:
            # nothing to replace
            return string


class ProtoConverterBase(ABC):

    @abstractmethod
    def convert(self, param, meta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ) -> Tuple:
        raise NotImplementedError('this interface is not implemented')
