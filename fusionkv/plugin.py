# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point for FusionKV (dense KV stack).

pyproject registers ``fusionkv = "fusionkv.plugin"`` under the
``vllm.general_plugins`` entry-point group. vLLM calls
``load_general_plugins()`` early in every process (API server, engine
core, workers), so this runs before any KVTransferConfig is resolved.

Two jobs:
1. register the FusionKV connector with KVConnectorFactory so
   ``--kv-transfer-config '{"kv_connector":"FusionKV",...}'`` resolves

2. apply elastic-mode patches (fusionkv.patch) when FUSIONKV_VA_BYTES is set.

Enable by adding "fusionkv" to VLLM_PLUGINS if plugin auto-loading is
restricted: ``VLLM_PLUGINS=fusionkv``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_registered = False
_installed = False


def register_connector() -> None:
    global _registered
    if _registered:
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except ImportError:  # vllm not importable in this process
        return
    if "FusionKV" in getattr(KVConnectorFactory, "_registry", {}):
        _registered = True
        return
    KVConnectorFactory.register_connector(
        "FusionKV", "fusionkv.connector", "FusionKVConnector")
    _registered = True
    logger.info("FusionKV: registered KV connector 'FusionKV' "
                "(fusionkv.connector:FusionKVConnector)")


def load() -> None:
    """General-plugin entry function (setuptools ``load`` convention)."""
    global _installed
    register_connector()
    if _installed:
        return
    _installed = True
    if os.environ.get("FUSIONKV_VA_BYTES", "").strip():
        # lazy: patch touches vllm internals; import vllm here is fine,
        # plugins load after vllm import
        from . import patch
        try:
            patch.install()
        except Exception:
            logger.exception(
                "FusionKV elastic mode failed to initialise; "
                "unset FUSIONKV_VA_BYTES to fall back to stock allocation")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load()

register = load  # bootstrap alias (zz_fuse_bootstrap tries plugin:register)
