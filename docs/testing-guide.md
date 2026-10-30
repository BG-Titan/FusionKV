# FusionKV 测试指南

> 适用版本：本仓库当前树（`fusionkv/` 0.1.0）。机制说明见
> [strata-kv-mechanism.md](strata-kv-mechanism.md)，总装说明见
> [../README.md](../README.md)。
>
> 测试分三层：**L1 CPU 单元测试**（无 GPU、无 torch 也能跑）→ **L2 GPU
> 原生冒烟**（要 GPU+torch，不要 vLLM）→ **L3 引擎端到端**（vLLM + 真实模型）。
> 逐层扩大置信域：纯逻辑 → 驱动 API/拷贝正确性 → 与调度器/模型的行为正确性。

## 0. 快速上手

```bash
cd fusionkv

# L1：任意开发机（Python >= 3.10，只需 pytest）
python -m pytest tests -q          # 期望：27 passed, 1 skipped

# L2 + L3：GPU 机器
pip install -e .                   # 会一并编译 C++ 原生核 fusionkv/_C*.so
python scripts/native_smoke.py     # VMM + 拷贝 parity 冒烟，见 §3
bash scripts/gpu_e2e.sh /path/to/model tier     # 或 elastic / both
```

## 1. 分层总览与验收标准

| 层 | 环境 | 耗时 | 覆盖 | 通过判据 |
|---|---|---|---|---|
| L1 单元 | 任意 + pytest | <1 s | 调度器协议、主机索引、拷贝计划、插件、配置 | `27 passed, 1 skipped`（无原生核时跳过 1 例是预期） |
| L2 原生冒烟 | GPU + torch | <2 min | libcuda dlopen、VMM 池、锁页注册、批量拷贝位一致、游程合并 | 物理映射随扩缩变化；round-trip 逐字节相等；连续 id 的 ops==1 |
| L3 端到端 | GPU + vLLM + 模型 | 10-30 min | 注册、greedy 一致性、主机层命中加速、弹性超售启动 | `ALL CHECKS PASSED`；见 §4 各检查断言 |

**顺序原则**：L1 常绿再动 Python 侧；L2 绿再动 C++/`worker.py` 拷贝路径；L3
是合入前与 GPU nightly 的门槛。

## 2. L1：CPU 单元测试（无 GPU / 无 torch）

### 2.1 运行

```bash
cd fusionkv
../.venv/bin/python -m pytest tests -q        # 解释器用绝对路径
```

> ⚠️ 若从 `fusionkv/` 目录用相对路径解释器（`../.venv/bin/python`）看到
> `RuntimeWarning: Unexpected value in sys.prefix`，是无害告警；用绝对路径
> （如 `/home/amd/dsh-work/.venv/bin/python`）即消失。

### 2.2 覆盖地图（测什么、为什么）

| 文件 | 用例 | 验证目标（对应机制/契约） |
|---|---|---|
| `tests/test_block_index.py` | 5 | 主机块索引生命周期：FREE→PENDING→READY；second-chance LRU 淘汰（`docs §1.4`）；pin 在 reserve 时存活、unpin 后可淘汰；链匹配只认 READY 前缀；`begin_store` 吸收重复哈希。`reserve` 全 pin 时抛 `IndexError('...pinned-full...')` |
| `tests/test_plan.py` | 9 | `runs_of` 两侧同递增才并游程（整前缀=1 条描述符，Strata `kv_stream.cu::runs_of`）；`copy_plan` 字节偏移/基址、跨步不等报错、`max_descriptor_bytes` 切分（含非整除上限）；`max_abs_delta` 有符号、按 min(len) 前缀 |
| `tests/test_scheduler_flow.py` | 8 | 用伪造 core 数据结构模拟调度器全契约：store→提交→二次会话命中；eager 边界 + resumed shadow 重建；lazy 只在容量压力时落盘 + 会话结束快照 + 延迟释放（`finished_sending` 注入协议）；抢占清账但保留在途 `_event_stores`（冲刷后仍提交）；TP 世界数聚合计数；load pin 在 finished_recving 释放且被 request_finished 尊重；孤儿命中释放 |
| `tests/test_connector_stub.py` | 4 | 用 `tests/stub_vllm.py` 顶替 `KVConnectorBase_V1`：调度器角色**无 torch 可导入**（导入卫生）；多 KV 组（混合 SWA）自动禁用分层走 pass-through；`hash_block_size` 细分块 `sub_per_fa=4` 链取样；哈希粒度比块粗时 `NotImplementedError` |
| `tests/test_plugin.py` | 3(+1 skip) | `register_connector` 幂等且可加载；未设 `FUSIONKV_VA_BYTES` 时 patch 不激活（默认路径零影响）；`patch.install()` 无原生核时报错（skip 判据即原生核存在与否） |

### 2.3 失败怎么读

| 症状 | 含义 | 去处 |
|---|---|---|
| `AssertionError: unbalanced unpin` | pin/unpin 记账不配对（一次 pin 恰一个持有者一次释放） | `fusionkv/block_index.py` `_pins` 计数；先查 `get_num_new_matched_tokens` 显式 pin 与 `_finish_load` 释放 |
| `drain_ready_to_free()` 返回空但应含 id | `request_finished` 未把未派发的 `_to_store` 项挪进 `state.tail`，或 `state.finished` 未置位 | `fusionkv/scheduler.py` `request_finished_all_groups` |
| stub 测试 `ImportError: vllm` | `tests/` 不在 `sys.path` | 从 `fusionkv/` 目录运行 pytest（conftest 不需要；文件头有 `sys.path.insert`） |
| lazy 用例队列永不清零 | 高水位判据 `free/num <= 1-high_water` 未触发 | 调小 `high_water` 或在用例里 `index.reserve` 制造压力 |

## 3. L2：GPU 原生冒烟（torch，不需要 vLLM）

### 3.1 安装与构建

```bash
pip install -e ./fusionkv
# 构建变量：FUSIONKV_CXX=g++ FUSIONKV_SKIP_BUILD=1 FUSIONKV_PY_INC=/path/to/Python.h目录
python -c "import fusionkv._C as C; print(C.driver_available())"   # True
```

`FUSIONKV_SKIP_BUILD=1` 或编译失败时模块照常工作——worker 自动退到
torch `index_select` 路径（功能等价、更慢），用例如需强制降级冒烟 fallback 即设它。

### 3.2 VMM 弹性池冒烟

原生核 dlopen libcuda，无静态 cudart 依赖。要点：粒度按驱动查询（通常 2 MiB），
`ensure_physical` 增映射、`shrink_physical` 按「高水位 + 尾部钉住块」收缩。

```python
import fusionkv._C as C
pool = C.VmmPool.create(va_bytes=8 << 30, phys_bytes=1 << 30, device_id=0)
s = pool.stats()            # dict(base, va_size, phys_mapped, chunk, ...)
assert s["phys_mapped"] >= 1 << 30
pool.ensure_physical(3 << 30)
assert pool.stats()["phys_mapped"] >= 3 << 30
pool.shrink_physical(1 << 30, 0, 1 << 20)   # target, pinned_tail_blocks, block_bytes
print(pool.stats())
```

引擎侧等价检查（无需手写脚本）：`fusionkv/elastic.py` 的 `VmmPool`/
`wrap_cuda_ptr` 会把池映射包成 torch 张量；`gpu_e2e.sh elastic` 模式末尾打印
`mapped X GiB of Y GiB VA` 就是它的成品形态。

### 3.3 拷贝位一致（parity）+ 游程合并冒烟

对应 Strata 的 `kv_stream_parity` 思路：D2H → 清零 → H2D → 逐字节比对，并验证
连续 id 只发 1 条 memcpy（游程合并生效）。

```python
import torch, fusionkv._C as C
BLK, N = 1 << 20, 16
gpu = torch.randint(0, 256, (N, BLK), dtype=torch.uint8, device="cuda")
host = torch.empty_like(gpu, device="cpu").pin_memory()
eng = C.CopyEngine()
eng.init_plane(gpu.data_ptr(), host.data_ptr(), BLK)
stream = torch.cuda.current_stream().cuda_stream
# D2H：块 3..10 连续 -> 应合并为 1 条 memcpy
ids = list(range(3, 11))
ops = eng.copy_batch(0, ids, ids, False, stream)   # direction False=D2H
torch.cuda.synchronize()
assert ops == 1, ops                                # 游程合并证据
assert torch.equal(gpu[3:11], host[3:11])
gpu[3:11] = 0
ops = eng.copy_batch(0, ids, ids, True, stream)     # H2D
torch.cuda.synchronize()
assert ops == 1 and torch.equal(gpu[3:11], host[3:11])
# 离散 id 不并游程；-1 视为驱动错误
assert eng.copy_batch(0, [0, 2, 5], [0, 2, 5], False, stream) == 3
torch.cuda.synchronize()
print("parity OK")
```

补充检查：离散 id（如 `[0, 2, 5]`）时 `ops == 3`（不并游程）；`copy_batch`
返回 `-1` 视为驱动错误直接失败。`C.host_register(ptr, size)` 重复注册按引用
计数安全（worker 用它给 arena 加注册）。§3.2 + §3.3 的成品可执行版：
`python scripts/native_smoke.py`（全过打印 `NATIVE SMOKE PASSED`）。

## 4. L3：引擎端到端（真实 GPU + vLLM）

### 4.1 前置：注册确认（检查 1）

三种生效路径任选其一，启动日志应出现 FusionKV connector/patch 安装行：

1. `pip install ./fusionkv` → `vllm.general_plugins` entry point 自动加载；
2. 源码目录 + `export VLLM_PLUGINS=fusionkv`（白名单环境必备）；
3. 不装插件：`kv_transfer_config` 里给 `kv_connector_module_path`
   （工厂支持按模块路径惰性加载）。

```bash
python -c "from fusionkv import plugin; plugin.register_connector(); \
import vllm.distributed.kv_transfer.kv_connector.v1 as k; \
assert 'FusionKV' in k.KVConnectorFactory._registry"
```

### 4.2 一键脚本 `scripts/gpu_e2e.sh`

```bash
bash scripts/gpu_e2e.sh /path/to/Qwen3-30B-A3B tier      # 默认
bash scripts/gpu_e2e.sh /path/to/model elastic
bash scripts/gpu_e2e.sh /path/to/model both
```

| 检查 | 做法 | 通过判据（脚本内断言） |
|---|---|---|
| 1 注册 | `python -c` 断言工厂登记表 | `'FusionKV' in registry` |
| 2 greedy 一致性 | 同一 prompt 分别用**纯 vLLM** 与 **FusionKV**（`enforce_eager, gpu_util=0.35`）生成 | 输出逐字符相等（connector 不改变任何 token） |
| 3 主机层命中 | `enable_prefix_caching=False` 跑 ~14k 字符 prompt → `llm.reset_prefix_cache()` 冲刷 GPU 前缀态 → 重放同 prompt | `second < first / 2`（命中来自主机层 H2D；若不过，看 `FUSIONKV_LOG=INFO` 的 store/hit 行） |
| 4 弹性启动 | `FUSIONKV_VA_BYTES=4x ADV, FUSIONKV_PHYS_INIT_BYTES=ADV/4, FUSIONKV_KV_BYTES=3x ADV, gpu_memory_utilization=0.25` 起引擎并生成 | `P.ELASTIC_POOL is not None` 且打印 `mapped < advertised` 的两行统计 |

全部通过打印 `ALL CHECKS PASSED (mode)`。**GPU 型号/模型无关**：dense GQA 与
MLA 均可；混合 SWA/EAGLE 模型只应看到 pass-through 告警且行为同纯 vLLM。

### 4.3 手动 A 层（分层）验证要点

```bash
VLLM_PLUGINS=fusionkv FUSIONKV_LOG=INFO FUSIONKV_CPU_BYTES=64MB \
vllm serve /path/to/model --gpu-memory-utilization 0.35 \
  --kv-transfer-config '{"kv_connector":"FusionKV","kv_role":"kv_both"}'
```

观察点（按时间顺序应出现）：

- 启动：host arena 分配（锁页）+ 「worker 侧对 GPU 层张量建 uint8 平面」；
- 会话结束：store 事件行（D2H，`store_scan_step`/高水位控制节奏）；
- 第二个相同前缀请求：**hit 行 + `WAITING_FOR_REMOTE_KVS`**，随后跳过重算；
- 压满 64 MB：second-chance 淘汰，无 pinned-full 报错重试风暴。

配置入口两类等价（`extra_config` 键 or 环境变量，见 §6）：
`cpu_bytes_to_use`、`high_water`、`store_batch_blocks`、`offload_after_secs`、
`lazy_store`、`store_scan_step`。

### 4.4 手动 B 层（弹性）验证要点

```bash
FUSIONKV_VA_BYTES=256GB FUSIONKV_PHYS_INIT_BYTES=8GB FUSIONKV_KV_BYTES=48GB \
FUSIONKV_LOG=INFO vllm serve ... --gpu-memory-utilization 0.25
```

判据：启动成功且调度器按 48 GB KV 预算规划；`nvidia-smi` 物理占用远低于
预算；随请求水位增长 `mapped` 上升（grow hint 由
`meta.scheduled_max_block_id` 送达 worker）；写越界不会发生——映射始终领先
已调度块号。收缩路径（`shrink_physical`）当前仅测试脚本调用，未接在线 trim。

### 4.5 回归矩阵（人工执行清单）

| # | 场景 | 操作 | 期望 |
|---|---|---|---|
| R1 | 抢占 | 低 `gpu_memory_utilization` + 高并发 | 无死锁；被抢占请求重跑成功；在途拷贝冲刷后仍提交（L1 有协议级用例） |
| R2 | 取消/超时 | 客户端中途断连 | `request_finished` 延迟释放生效，主机内存不涨漏（RSS 稳态） |
| R3 | 超长 prompt | 单请求 > KV 池块数 | 与纯 vLLM 一致（拒绝或分块，视配置）；FusionKV 不新增失败模式 |
| R4 | 重启 | 杀进程重起 | 主机层是**进程内**层：冷启动全 miss 属预期（跨进程持久化属 KVS/NIXL 场景，非 v1 范围） |
| R5 | TP>1 | `--tensor-parallel-size 2` | 每 rank 独立 arena；`cpu_bytes_to_use` 按 world_size 均分（或 `cpu_bytes_to_use_per_rank` 显式覆盖）；store 完成计数聚合到 world_size 才提交 |
| R6 | 混合模型 | gemma 系（SWA+full）/EAGLE | 告警 `multi-group... passthrough`，行为与纯 vLLM 一致 |
| R7 | 默认路径零影响 | 不配 kv_transfer_config、不设 FUSIONKV_* | 引擎行为与未装 FusionKV 逐 token 一致（检查 2 的退化形式） |

## 5. 性能观测建议（非门禁，看数）

- **命中收益**：检查 3 的 `first/second` 比；生产口径 = TTFT 分位数对比。
- **拷贝效率**：L2 parity 的 `ops` 数即游程效果；整前缀恢复应为
  每平面 1 条 DMA（`plan.py` docstring 的 Strata 观察）。
- **带宽**：D2H GB/s ≈ arena 字节 / store 事件间隔；H2D 命中期与计算重叠
  （专用流 + `WAITING_FOR_REMOTE_KVS` 门控，不阻塞其它请求）。
- **弹性水位**：`stats()["phys_mapped"]` 曲线 vs 时间，验证「先映射后使用」
  的领先量，若频繁贴边则调大 `FUSIONKV_PHYS_INIT_BYTES`。

## 6. 环境变量 / 配置速查

| 键 | 默认 | 作用 |
|---|---|---|
| `FUSIONKV_CPU_BYTES`（`cpu_bytes_to_use`） | 0=关 | 主机锁页 arena 总大小（`64M/8G` 后缀按 MiB/GiB） |
| `FUSIONKV_CPU_BYTES_PER_RANK` | — | 覆盖按 world_size 的均分 |
| `FUSIONKV_HIGH_WATER` | 0.9 | lazy 落盘触发的空闲占比阈值（≤1-x 即触发） |
| `FUSIONKV_STORE_BATCH_BLOCKS` | 512 | 每次 store 事件最多块数 |
| `FUSIONKV_STORE_SCAN_STEP` | 1024 | 扫描/入队步长 |
| `FUSIONKV_LAZY_STORE` | false | 只容量压力落盘（否则会话结束快照也存） |
| `FUSIONKV_OFFLOAD_AFTER_SECS` | 0 | 空闲 N 秒后才落盘 |
| `FUSIONKV_VA_BYTES` / `FUSIONKV_PHYS_INIT_BYTES` | 0=关 | B 层：VA 预留 / 初始物理映射 |
| `FUSIONKV_KV_BYTES` | — | B 层：向调度器广播的 KV 预算（>物理即超售） |
| `FUSIONKV_RING_BYTES` | — | 设备拷贝环 staging 大小 |
| `FUSIONKV_LOG` | — | `DEBUG/INFO/WARNING`，只开 fusionkv.* 日志 |
| `VLLM_PLUGINS` | — | vLLM 插件白名单，需含 `fusionkv` |
| `FUSIONKV_SKIP_BUILD` / `FUSIONKV_CXX` / `FUSIONKV_PY_INC` | — | 构建控制（§3.1） |

## 7. 故障排查表

| 症状 | 原因 | 处置 |
|---|---|---|
| 启动报 connector `FusionKV` not found | 插件未加载 | 三选一（§4.1）；pip 装过又改源码要 `pip install -e` 或确认 entry point 指向已改代码 |
| `ValueError: could not convert string to float: '64M'` | 旧版 parse_bytes 后缀序 bug | 已修（KIB/MB… 序）；拉最新 `config.py` |
| `No module named fusionkv._C` | 未编译/跳过编译 | 正常降级可用；要原生路径见 §3.1（`FUSIONKV_PY_INC` 指头文件） |
| 命中恒 0（检查 3 不过） | GPU 前缀缓存没冲干净 / store 未发生 | 确认 `enable_prefix_caching=False` 且 `reset_prefix_cache()`；`FUSIONKV_LOG=INFO` 看 store 提交行；检查两次请求前缀逐字符一致 |
| 命中恒 0（真实流量） | `hash_block_size` 与块粒度不齐（链取样失败）或 `skip_reading_prefix_cache` | L1 有粒度用例；日志查 warn_once |
| `fusionkv host index is pinned-full; retry later` | 全部 slot 被在途引用 pin | 正常背压（调度器重试）；持续出现则调大 arena 或降 `store_batch_blocks` |
| elastic：`KV planes outside pool` 类告警 | patch 未激活（未设 `FUSIONKV_VA_BYTES` 或插件顺序在 vLLM 分配之后） | 用 §4.1 路径 1/2 保证 `plugin.load()` 先于 worker 初始化 |
| elastic 运行中写越界崩溃 | 初始物理 < capture 余量且 grow hint 未到 | 调大 `FUSIONKV_PHYS_INIT_BYTES`（capture 图会预触页） |
| `mlock`/pin 失败 | ulimit 限制 | `ulimit -l unlimited` 或减小 `FUSIONKV_CPU_BYTES` |
| 多组/EAGLE 模型无分层 | v1 保守禁用（正确行为） | 看 `passthrough` 告警确认；勿当 bug 报 |

## 8. CI 建议

- **每次 PR（headless，<1 s）**：`python -m pytest tests -q`，门禁 L1 全绿。
- **合入前 / GPU nightly**：`gpu_e2e.sh both` 一套全过；另跑 R5（TP2）一次。
- **改 `csrc/` 或 `worker.py` 拷贝路径时**：必跑 §3.3 parity + 检查 2。
- 测试矩阵建议：{dense GQA, MLA} × {tier, elastic} 各一模型即可覆盖主干；
  混合模型跑「行为一致性」（R6）即可。
- 用例卫生：新增 CPU 用例**禁止**顶层 `import torch`（调度器角色可在 CI
  无 torch 环境跑）——`tests/stub_vllm.py` 已提供基类替身。
