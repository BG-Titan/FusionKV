# SPDX-License-Identifier: Apache-2.0
"""FusionKV: Strata-inspired tiered + elastic KV cache for vLLM.

External module (pip package + vLLM general-plugin) implementing:

* A-tier: pinned host KV arena managed by a C++ core (dlopen'd CUDA driver,
  no cuda-python dependency), run-coalesced batch D2H/H2D, LRU + second-chance
  host block index, async load via ``WAITING_FOR_REMOTE_KVS``.
* B-elastic: VMM-backed GPU pool - reserve huge VA, map physical 2 MiB chunks
  on demand (Strata ``kv_reserve``), plus a device copy-ring staging pool
  (Strata ``kv_alloc``) so the engine's VRAM floor drops below one full
  context window.

Layout:
  fusionkv/config.py      env + extra-config parsing
  fusionkv/wire.py        KVConnectorMetadata / worker metadata
  fusionkv/plan.py        pure copy-plan run coalescing (mirrors Strata runs_of)
  fusionkv/block_index.py pure host block index (LRU + second chance)
  (pinned host arena lives in worker.py: torch pin_memory + native
   host_register; see csrc/pin_arena.cpp for the registration core)
  fusionkv/elastic.py     VMM pool python wrapper (adopt + grow hint)
  fusionkv/_C*.so         native core (csrc/*.cpp, dlopen libcuda; optional)
  fusionkv/worker.py      worker-side scheduler state + copy orchestration
  fusionkv/scheduler.py   scheduler-side manager (hits, stores, metadata)
  fusionkv/connector.py   KVConnectorBase_V1 glue (registered as "FusionKV")
  fusionkv/plugin.py      vllm.general_plugins entry point
  fusionkv/patch.py       startup patches for elastic mode (no vLLM edits)
"""

import logging
import os

FUSIONKV_CONNECTOR_NAME = "FusionKV"

# FUSIONKV_LOG=INFO|DEBUG turns on fusionkv.* logging without touching vLLM's.
_LEVEL = (os.environ.get("FUSIONKV_LOG") or
            os.environ.get("FUSIONKV_LOG") or "").upper()
if _LEVEL in ("DEBUG", "INFO", "WARNING"):
    _root = logging.getLogger("fusionkv")
    _root.setLevel(_LEVEL)
    if not any(isinstance(h, logging.StreamHandler) for h in _root.handlers):
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(
            "fusionkv %(levelname)s %(name)s: %(message)s"))
        _root.addHandler(_h)
        _root.propagate = False

__all__ = ["FUSIONKV_CONNECTOR_NAME"]
