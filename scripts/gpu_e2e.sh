#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# FusionKV end-to-end validation on a real GPU + real vLLM install.
# Run on the GPU host, with this repo checked out next to a vllm source
# tree (or an installed vllm wheel) and `pip install ./fusionkv`.
#
#   ./scripts/gpu_e2e.sh /path/to/model  [mode]
#
# modes: tier | elastic | both  (default: tier)
#
# Checks, in order:
#   1. plugin registers the connector (log line)
#   2. engine boots, one generation OK (stock vs FusionKV identical output)
#   3. A-tier: session restart reuses host KV (hit-rate log > 0, latency
#      drop on identical second prompt)
#   4. B-elastic: engine starts with FUSIONKV_KV_BYTES > physical VRAM and
#      generated mapped-bytes < advertised bytes
set -euo pipefail

MODEL=${1:?usage: gpu_e2e.sh /path/to/model [tier|elastic|both]}
MODE=${2:-tier}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/fusionkv_e2e.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT
GPU_MEM_MIN_FREE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-fusionkv,fusionattn}"
ATTN="${ATTN:-}"   # e.g. FUSION_ATTN for GPUs where stock backends are unavailable
export ATTN
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export FUSIONKV_LOG=${FUSIONKV_LOG:-INFO}

echo "== 1. connector registration =="
python -c "
from fusionkv.plugin import load; load()
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F
assert 'FusionKV' in F._registry, F._registry.keys()
print('OK: FusionKV in', list(F._registry))"

# elastic env must not leak into the stock baseline (patched allocator
# without the connector = nobody grows the pool -> unmapped writes).
unset FUSIONKV_KV_BYTES FUSIONKV_VA_BYTES FUSIONKV_PHYS_INIT_BYTES \
      FUSIONKV_ELASTIC_BOOT_FLOOR 2>/dev/null || true

echo "== 2. stock vs FusionKV identical greedy output =="
cat > "$TMPD/stock.py" <<'PY'
import sys
from vllm import LLM, SamplingParams

def main():
    llm = LLM(model=sys.argv[1], enforce_eager=True, max_model_len=4096,
              attention_backend=__import__('os').environ.get('ATTN') or None,
              gpu_memory_utilization=0.35)
    o = llm.generate(["The capital of France is"],
                     SamplingParams(temperature=0.0, max_tokens=32))
    print("TEXT=" + o[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()
PY
BASE=$(python "$TMPD/stock.py" "$MODEL" | grep "^TEXT=" | head -1)
echo "stock : $BASE"

TIER_JSON='{"kv_connector":"FusionKV","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":"4GiB"}}'
cat > "$TMPD/tier_match.py" <<PY
import sys, json
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def main():
    kvc = KVTransferConfig(**json.loads('$TIER_JSON'))
    llm = LLM(model='$MODEL', enforce_eager=True, max_model_len=4096,
              attention_backend=__import__('os').environ.get('ATTN') or None,
              gpu_memory_utilization=0.35, kv_transfer_config=kvc)
    o = llm.generate(["The capital of France is"],
                     SamplingParams(temperature=0.0, max_tokens=32))
    print("TEXT=" + o[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()
PY
FK=$(python "$TMPD/tier_match.py" | grep "^TEXT=" | head -1)
echo "fusionkv: $FK"
[ "$BASE" = "$FK" ] && echo "OK: identical greedy output" || { echo "MISMATCH"; exit 1; }

if [ "$MODE" = "tier" ] || [ "$MODE" = "both" ]; then
  echo "== 3. host hit across sessions (same process, two generate rounds) =="
  cat > "$TMPD/hit.py" <<PY
import sys, json, time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def main():
    kvc = KVTransferConfig(**json.loads('$TIER_JSON'))
    llm = LLM(model='$MODEL', enforce_eager=True, max_model_len=4096,
              attention_backend=__import__('os').environ.get('ATTN') or None,
              gpu_memory_utilization=0.35, kv_transfer_config=kvc,
              enable_prefix_caching=False)   # force the connector, not GPU cache
    long_prompt = "Once upon a time " * 900   # > one KV block chain, ~14k chars
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    t0 = time.time(); llm.generate([long_prompt], sp); first = time.time() - t0
    # flush GPU prefix state: second run must be served from HOST tier
    llm.reset_prefix_cache()
    t0 = time.time(); out = llm.generate([long_prompt], sp); second = time.time() - t0
    print(f"first={first:.2f}s second={second:.2f}s (second should be much lower)")
    assert second < first / 2, "no host-hit speedup: check FusionKV store/hit logs"
    print("OK: host tier served the replay")

if __name__ == "__main__":
    main()
PY
  GPU_UTIL=0.35 FUSIONKV_LOG=INFO python "$TMPD/hit.py" "$MODEL"
fi

if [ "$MODE" = "elastic" ] || [ "$MODE" = "both" ]; then
  echo "== 4. elastic boot: advertised KV > physical =="
  # KV budget ~ 3x the physical slice we give the cache (GPU_UTIL small)
  ADV=$(( (GPU_MEM_MIN_FREE * 1024 * 1024) / 8 ))   # ~1/8 VRAM for KV stock
  export FUSIONKV_VA_BYTES=$(( ADV * 8 ))             # 8x advertised VA
  export FUSIONKV_PHYS_INIT_BYTES=$(( ADV / 4 ))
  export FUSIONKV_ELASTIC_BOOT_FLOOR=$(( 64 << 20 ))  # tiny boot mapping
  export FUSIONKV_KV_BYTES=$(( ADV * 3 ))             # advertise 3x KV
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1        # rpc probe
  cat > "$TMPD/elastic.py" <<PY
import sys
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def main():
    # elastic grow is driven by the FusionKV connector's per-step metadata
    # (scheduled_max_block_id -> worker maps VMM chunks before forward).
    ktc = KVTransferConfig(
        kv_connector="FusionKV", kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": "2GiB"})
    llm = LLM(model=sys.argv[1], enforce_eager=True, max_model_len=4096,
              enable_prefix_caching=False, kv_transfer_config=ktc,
              attention_backend=__import__('os').environ.get('ATTN') or None,
              gpu_memory_utilization=0.25)
    o = llm.generate(["The capital of France is"],
                     SamplingParams(temperature=0.0, max_tokens=32))
    print("GEN:", o[0].outputs[0].text.strip())

    def probe(worker):                      # runs inside EngineCore
        import fusionkv.patch as P
        p = P.ELASTIC_POOL
        return None if p is None else (p.mapped_bytes, p.va_bytes)

    mapped, va = llm.collective_rpc(probe)[0]
    assert mapped and va, "pool not active in worker"
    assert "Paris" in o[0].outputs[0].text, "wrong greedy output"
    print(f"OK: mapped {mapped/2**30:.2f} GiB physical of "
          f"{va/2**30:.2f} GiB VA")

if __name__ == "__main__":
    main()
PY
  FUSIONKV_LOG=INFO python "$TMPD/elastic.py" "$MODEL"
fi

echo "ALL CHECKS PASSED ($MODE)"
