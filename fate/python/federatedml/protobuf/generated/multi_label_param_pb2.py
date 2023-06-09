# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: multi-label-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17multi-label-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"-\n\x0cLabelMapping\x12\r\n\x05label\x18\x01 \x01(\t\x12\x0e\n\x06mapped\x18\x02 \x01(\t\"\xcc\x01\n\x14MultiLabelModelParam\x12\x16\n\x0e\x61ggregate_iter\x18\x01 \x01(\x05\x12\x14\n\x0closs_history\x18\x02 \x03(\x01\x12\x14\n\x0cis_converged\x18\x03 \x01(\x08\x12\x0e\n\x06header\x18\x04 \x03(\t\x12K\n\rlabel_mapping\x18\x05 \x03(\x0b\x32\x34.com.webank.ai.fate.core.mlmodel.buffer.LabelMapping\x12\x13\n\x0b\x61pi_version\x18\x06 \x01(\rB\x16\x42\x14MultiLabelParamProtob\x06proto3')



_LABELMAPPING = DESCRIPTOR.message_types_by_name['LabelMapping']
_MULTILABELMODELPARAM = DESCRIPTOR.message_types_by_name['MultiLabelModelParam']
LabelMapping = _reflection.GeneratedProtocolMessageType('LabelMapping', (_message.Message,), {
  'DESCRIPTOR' : _LABELMAPPING,
  '__module__' : 'multi_label_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.LabelMapping)
  })
_sym_db.RegisterMessage(LabelMapping)

MultiLabelModelParam = _reflection.GeneratedProtocolMessageType('MultiLabelModelParam', (_message.Message,), {
  'DESCRIPTOR' : _MULTILABELMODELPARAM,
  '__module__' : 'multi_label_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.MultiLabelModelParam)
  })
_sym_db.RegisterMessage(MultiLabelModelParam)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\024MultiLabelParamProto'
  _LABELMAPPING._serialized_start=67
  _LABELMAPPING._serialized_end=112
  _MULTILABELMODELPARAM._serialized_start=115
  _MULTILABELMODELPARAM._serialized_end=319
# @@protoc_insertion_point(module_scope)
