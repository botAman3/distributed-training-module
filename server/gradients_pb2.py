# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: gradients.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'gradients.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fgradients.proto\x12\tgradients\"7\n\x08Gradient\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x03(\x02\x12\r\n\x05shape\x18\x03 \x03(\x05\"Z\n\x0eGradientPacket\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12&\n\tgradients\x18\x02 \x03(\x0b\x32\x13.gradients.Gradient\x12\r\n\x05\x65poch\x18\x03 \x01(\x05\"\"\n\rStartTraining\x12\x11\n\tclient_id\x18\x01 \x01(\t\"e\n\rServerMessage\x12\x15\n\x0binstruction\x18\x01 \x01(\tH\x00\x12\x36\n\x11updated_gradients\x18\x02 \x01(\x0b\x32\x19.gradients.GradientPacketH\x00\x42\x05\n\x03msg\"p\n\rClientMessage\x12(\n\x04join\x18\x01 \x01(\x0b\x32\x18.gradients.StartTrainingH\x00\x12.\n\tgradients\x18\x02 \x01(\x0b\x32\x19.gradients.GradientPacketH\x00\x42\x05\n\x03msg2^\n\x0fGradientService\x12K\n\x11\x46\x65\x64\x65ratedTraining\x12\x18.gradients.ClientMessage\x1a\x18.gradients.ServerMessage(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'gradients_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_GRADIENT']._serialized_start=30
  _globals['_GRADIENT']._serialized_end=85
  _globals['_GRADIENTPACKET']._serialized_start=87
  _globals['_GRADIENTPACKET']._serialized_end=177
  _globals['_STARTTRAINING']._serialized_start=179
  _globals['_STARTTRAINING']._serialized_end=213
  _globals['_SERVERMESSAGE']._serialized_start=215
  _globals['_SERVERMESSAGE']._serialized_end=316
  _globals['_CLIENTMESSAGE']._serialized_start=318
  _globals['_CLIENTMESSAGE']._serialized_end=430
  _globals['_GRADIENTSERVICE']._serialized_start=432
  _globals['_GRADIENTSERVICE']._serialized_end=526
# @@protoc_insertion_point(module_scope)
