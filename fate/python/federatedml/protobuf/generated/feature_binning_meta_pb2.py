# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-binning-meta.proto
<<<<<<< HEAD

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
=======
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




<<<<<<< HEAD
DESCRIPTOR = _descriptor.FileDescriptor(
  name='feature-binning-meta.proto',
  package='com.webank.ai.fate.core.mlmodel.buffer',
  syntax='proto3',
  serialized_options=_b('B\027FeatureBinningMetaProto'),
  serialized_pb=_b('\n\x1a\x66\x65\x61ture-binning-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"?\n\rTransformMeta\x12\x16\n\x0etransform_cols\x18\x01 \x03(\x03\x12\x16\n\x0etransform_type\x18\x02 \x01(\t\"\xa3\x02\n\x12\x46\x65\x61tureBinningMeta\x12\x10\n\x08need_run\x18\x01 \x01(\x08\x12\x0e\n\x06method\x18\n \x01(\t\x12\x16\n\x0e\x63ompress_thres\x18\x02 \x01(\x03\x12\x11\n\thead_size\x18\x03 \x01(\x03\x12\r\n\x05\x65rror\x18\x04 \x01(\x01\x12\x0f\n\x07\x62in_num\x18\x05 \x01(\x03\x12\x0c\n\x04\x63ols\x18\x06 \x03(\t\x12\x19\n\x11\x61\x64justment_factor\x18\x07 \x01(\x01\x12\x12\n\nlocal_only\x18\x08 \x01(\x08\x12N\n\x0ftransform_param\x18\t \x01(\x0b\x32\x35.com.webank.ai.fate.core.mlmodel.buffer.TransformMeta\x12\x13\n\x0bskip_static\x18\x0b \x01(\x08\x42\x19\x42\x17\x46\x65\x61tureBinningMetaProtob\x06proto3')
)




_TRANSFORMMETA = _descriptor.Descriptor(
  name='TransformMeta',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.TransformMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='transform_cols', full_name='com.webank.ai.fate.core.mlmodel.buffer.TransformMeta.transform_cols', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_type', full_name='com.webank.ai.fate.core.mlmodel.buffer.TransformMeta.transform_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=133,
)


_FEATUREBINNINGMETA = _descriptor.Descriptor(
  name='FeatureBinningMeta',
  full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='need_run', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.need_run', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='method', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.method', index=1,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='compress_thres', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.compress_thres', index=2,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='head_size', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.head_size', index=3,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.error', index=4,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bin_num', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.bin_num', index=5,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cols', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.cols', index=6,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='adjustment_factor', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.adjustment_factor', index=7,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='local_only', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.local_only', index=8,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_param', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.transform_param', index=9,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='skip_static', full_name='com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta.skip_static', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=427,
)

_FEATUREBINNINGMETA.fields_by_name['transform_param'].message_type = _TRANSFORMMETA
DESCRIPTOR.message_types_by_name['TransformMeta'] = _TRANSFORMMETA
DESCRIPTOR.message_types_by_name['FeatureBinningMeta'] = _FEATUREBINNINGMETA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

=======
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1a\x66\x65\x61ture-binning-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"?\n\rTransformMeta\x12\x16\n\x0etransform_cols\x18\x01 \x03(\x03\x12\x16\n\x0etransform_type\x18\x02 \x01(\t\"\xc2\x02\n\x12\x46\x65\x61tureBinningMeta\x12\x10\n\x08need_run\x18\x01 \x01(\x08\x12\x0e\n\x06method\x18\n \x01(\t\x12\x16\n\x0e\x63ompress_thres\x18\x02 \x01(\x03\x12\x11\n\thead_size\x18\x03 \x01(\x03\x12\r\n\x05\x65rror\x18\x04 \x01(\x01\x12\x0f\n\x07\x62in_num\x18\x05 \x01(\x03\x12\x0c\n\x04\x63ols\x18\x06 \x03(\t\x12\x19\n\x11\x61\x64justment_factor\x18\x07 \x01(\x01\x12\x12\n\nlocal_only\x18\x08 \x01(\x08\x12N\n\x0ftransform_param\x18\t \x01(\x0b\x32\x35.com.webank.ai.fate.core.mlmodel.buffer.TransformMeta\x12\x13\n\x0bskip_static\x18\x0b \x01(\x08\x12\x1d\n\x15optimal_metric_method\x18\x0c \x01(\tB\x19\x42\x17\x46\x65\x61tureBinningMetaProtob\x06proto3')



_TRANSFORMMETA = DESCRIPTOR.message_types_by_name['TransformMeta']
_FEATUREBINNINGMETA = DESCRIPTOR.message_types_by_name['FeatureBinningMeta']
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
TransformMeta = _reflection.GeneratedProtocolMessageType('TransformMeta', (_message.Message,), {
  'DESCRIPTOR' : _TRANSFORMMETA,
  '__module__' : 'feature_binning_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.TransformMeta)
  })
_sym_db.RegisterMessage(TransformMeta)

FeatureBinningMeta = _reflection.GeneratedProtocolMessageType('FeatureBinningMeta', (_message.Message,), {
  'DESCRIPTOR' : _FEATUREBINNINGMETA,
  '__module__' : 'feature_binning_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta)
  })
_sym_db.RegisterMessage(FeatureBinningMeta)

<<<<<<< HEAD

DESCRIPTOR._options = None
=======
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\027FeatureBinningMetaProto'
  _TRANSFORMMETA._serialized_start=70
  _TRANSFORMMETA._serialized_end=133
  _FEATUREBINNINGMETA._serialized_start=136
  _FEATUREBINNINGMETA._serialized_end=458
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
# @@protoc_insertion_point(module_scope)
