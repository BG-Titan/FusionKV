# SPDX-License-Identifier: Apache-2.0
"""Plugin entry behavior without vllm/torch installed."""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import stub_vllm  # noqa: E402


def test_register_and_load(tmp_path, monkeypatch):
    stub_vllm.install()
    # minimal factory module stub
    reg = {}

    class F:
        _registry = reg

        @classmethod
        def register_connector(cls, name, module_path, class_name):
            reg[name] = (module_path, class_name)

    fac = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.factory")
    fac.KVConnectorFactory = F
    sys.modules[fac.__name__] = fac

    os.environ.pop("FUSIONKV_VA_BYTES", None)
    import fusionkv.plugin as plugin
    plugin._registered = False
    plugin._installed = False
    plugin.load()
    assert reg == {"FusionKV": ("fusionkv.connector", "FusionKVConnector")}
    plugin.load()                      # idempotent
    assert len(reg) == 1
    # without FUSIONKV_VA_BYTES the elastic patch module is not even imported
    assert "fusionkv.patch" not in sys.modules or \
        sys.modules["fusionkv.patch"].ELASTIC_POOL is None


def test_patch_inactive_without_env(monkeypatch):
    os.environ.pop("FUSIONKV_VA_BYTES", None)
    from fusionkv import patch
    assert patch.elastic_enabled() is False
    assert patch.install() is False
    assert patch.ELASTIC_POOL is None


def test_patch_requires_native(monkeypatch):
    monkeypatch.setenv("FUSIONKV_VA_BYTES", "8GiB")
    from fusionkv import patch
    if patch.HAVE_NATIVE:
        pytest.skip("native present; failure path not reachable")
    with pytest.raises(RuntimeError):
        patch._PATCHED = False
        patch.install()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
