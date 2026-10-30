# SPDX-License-Identifier: Apache-2.0
"""Minimal stubs to import fusionkv.connector without torch/vllm installed.

Used by tests/test_connector_stub.py. Mirrors the real API shapes
(vllm/v1/kv_cache_interface.KVCacheGroupSpec etc.), not behaviours.
"""
import sys
import types


class KVConnectorRole:
    SCHEDULER = 0
    WORKER = 1


class KVConnectorBase_V1:
    """Mirrors the real base __init__ attribute names (base.py:~240)."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self._connector_metadata = None
        self._vllm_config = vllm_config
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._kv_cache_config = kv_cache_config
        self._kv_cache_manager = None
        self._role = role


class SupportsHMA:
    pass


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install():
    _stub("vllm")
    _stub("vllm.config")
    _stub("vllm.distributed")
    _stub("vllm.distributed.kv_transfer")
    _stub("vllm.distributed.kv_transfer.kv_connector")
    _stub("vllm.distributed.kv_transfer.kv_connector.v1")
    _stub("vllm.distributed.kv_transfer.kv_connector.v1.base",
          KVConnectorBase_V1=KVConnectorBase_V1,
          KVConnectorRole=KVConnectorRole, SupportsHMA=SupportsHMA,
          KVConnectorMetadata=object)
    _stub("vllm.logger", logger=types.SimpleNamespace(
        info=lambda *a: None, warning=lambda *a: None,
        exception=lambda *a: None))
    _stub("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
    return KVConnectorRole
