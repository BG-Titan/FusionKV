# Strata 单会话 KV 显存机制完整解析 —— FusionKV 的设计依据

> 命名说明：对外组件名已迁移为 FusionAttention/FusionKV（稠密）与 FissionAttention/FissionKV（MoE），旧名 FUSION_ATTN/FUSE_MLA/FusionKV/FusionKV 保留为兼容别名；详见 docs/fusion-fission-naming.md。

> 对象：`/home/amd/dsh-work/Strata`（单会话推理引擎，CUDA/HIP，GGUF MoE）。
> 目的：完整解释 Strata 为什么能在小显存上跑 131K~262K 长上下文的 KV，
> 并给出每种机制到 vLLM 外置模块 FusionKV 的映射。所有结论均来自源码与仓库文档，
> 文末附文件行号索引。实现与配置见 [`../README.md`](../README.md)，
> 验证方法见 [testing-guide.md](testing-guide.md)。

## 0. 一句话总览

Strata 把「KV 必须全在显存」拆成了三个正交机制，可以叠加使用：

| 机制 | 开关 | 核心思想 | 显存效果 |
|---|---|---|---|
| **A. 冷热分层（streaming）** | `--kv-resident N` | 显存只保留 **N 个 cell 的热窗口**（驻留池），**全部 cell 的权威 KV 放在锁页内存**；注意力引用到不在热窗口的块时，当场从内存拷入并逐出冷块 | KV 显存从 O(上下文) 降为 O(窗口 N)，与上下文长度无关 |
| **B. VMM 弹性池（elastic）** | `--kv-grow` | KV 虚拟地址一次_reserve_到全上下文，物理页按 2 MiB chunk **到达才映射**；chunk 与专家缓存互相借还 | KV 显存 = 实际到达过的 cell 数；指针不变，CUDA graph/内核零感知 |
| **C. 专家缓存分段（同一思想的对偶）** | `--vram-elastic` + `POST /v1/vram` | 专家缓存分段映射，运行时按 `reserve_mib` 让出/收回显存 | 对外 API：给别的程序腾显存 |

A 解决「单会话 KV 总量 ≫ 显存」；B 解决「启动就为一整条上下文付 KV 显存，哪怕永远跑不到」；
B 与 C 共享同一套 VMM 基础设施（`include/strata/core/vmm.hpp`：`cuMemAddressReserve` +
每 2 MiB granularity 一个 `CUmemGenericAllocationHandle` chunk）。`--vram-elastic` 会关掉
`--kv-grow`（两者同时要求时整 KV 常驻显存，见 `docs/VRAM_ELASTIC.md` 末段）。

---

## 1. 机制 A：单会话冷热窗口（`--kv-resident`，本文重点）

### 1.1 三种 KV 布局（`kv_plan`，src/core/layer.cpp:738）

每个注意力 state 的 KV 池按 `g_kv_resident`（`--kv-resident`，下限
`qsa_kv_resident_min() = 20480` cell，layer.cpp:771）规划成三种模式：

- **mode 0（全常驻）**：`--kv-resident` 未开或窗口 ≥ 上下文。slots = pages，page table 恒等。
  这是默认路径，数值不变。
- **mode 1（主层热窗口）**：显存 slots = `ceil(max(N, 20480)/page_size)` 个页槽；
  上下文的全部 pages 的 KV 由**锁页内存镜像**承载（权威拷贝）。
- **mode 2（drafter 环）**：MTP 草稿头没有 indexer，KV 用**固定环**：初始 page table 即
  `block % n_slots`（`kv_ring_table`，layer.cpp:953 处初始化），写满自动覆盖最老槽位，
  不需要 miss/逐出 machinery。环从主机拷贝恢复后读出值与常驻池一致（parity 测试第 4 条，
  `src/kernels/kv_stream_parity.cpp:10`）。

关键不变量：**注意力永远只读显存槽；权威数据在内存**。所以热窗口是"缓存"，
内存镜像是"磁盘"，二者的块内容逐位相同 ⇒ 数值与全常驻 **bitwise** 一致
（`kv_stream_parity.cpp:7`：streamed vs resident 输出按位相等）。

### 1.2 数据结构（GPU 侧，均为 int32）

`KvStreamMap`（mode 1 才分配，layer.cpp:868-884）：

```
page_table[pages]   cell 块 → 显存槽，-1 = 不驻留      ← vLLM block table 的同款间接层
slot_block[slots]   槽 → 现在放的是哪个块（epoch 编码在 stamp）
slot_stamp[slots]   槽内数据的 epoch；reset 填 INT_MAX 表示空槽
slot_ref[slots]     second-chance 引用位（时钟扫描用）
miss_block/miss_slot[slots]  本轮 resolve 记下的缺口（每轮 ≤ slots 个，页表互异性保证
                             缺口不会多于槽数，kv_stream.cu 头注）
ctl[kKvCtlInts=128] epoch, sweep, nmiss, nres, cut, 以及 s_nmiss/s_lookups/s_cut 计数器
```

锁页内存镜像（layer.cpp:900-940）：`cudaHostAlloc(Mapped|Portable)` 分配
`pages × kv_block_bytes(格式)` 的**恒等布局**镜像，`cudaHostGetDevicePointer` 取设备指针，
内核直接经 PCIe 读。「恒等布局」= 块 b 的字节 run 固定为 `[b*len, (b+1)*len)`，
拷贝不需要任何索引（1.4 的 `runs_of`）。

镜像按**量化格式**存（与显存池同格式，`runs_of`，kv_stream.cu:31-65）：
fp16、int8（K/V codes + 每组 2 字节 scale）、Q4_0、K8V4 混合三种布局各自成套。
收益：① PCIe 流量缩到 1/2~1/4；② 池与镜像逐位同格式 ⇒ 换入换出**无重量化** ⇒ 位一致。

### 1.3 每轮注意力的三步（src/kernels/cuda/kv_stream.cu 头注，全文件仅 286 行）

```
resolve_kernel（1 个 1024 线程 block）
  ├─ 命中：page_table[id] 有效 → 直接解析出槽
  └─ 缺：块级 first-touch 登记（block_scan 前缀和，每块至多登记一次），
     victim = 时钟扫描(second chance)：从 ctl.sweep 起找第一个 ref==0 的槽；
     扫到 ref==1 就清零并前进；一圈内必有 ref==0（每步开始 reset 清零），
     一圈都没找到才允许回绕；stamp≠epoch 的空槽天然可选。
     victim 的旧块在 page_table 里改回 -1。
copy_kernel（每缺口 1 block，grid-stride）
  └─ 从锁页镜像按 runs_of 的字节 run 整块拷入新槽（16B 向量化，4 个 run 链式推进）。
reset_kernel（回合尾）
  └─ 清引用位、把本轮登记的块重新标为驻留（其 run 可能整块拷入过序列尾部之外的 cell，
     整块拷贝使这些邻居 cell 一并变驻留，无需再拷）。
```

- **写路径**：新 token 的 KV append 时，量化内核每半个 kernel 顺手写主机镜像
  （layer.cpp:1211 "each half also writes its part of the host copy"，K8V4 版
  `kv_hybrid_write_host`）。⇒ 内存镜像始终是全 cell 权威。
- **读路径**：注意力查询先经 resolve（layer.cpp:1261 "every block the selection names is
  made resident before anything reads it"）——DSA 的 index 选择已经把每步要读的块限制成
  top-k 稀疏集，所以每步缺口有限、拷贝量有上界（≤ slots 个块/轮）。
- **回合结束/新会话**：`kv_stream_reset` 把显存全部标为不驻留，但**不清主机镜像**
  （layer.cpp:989：清一次要过几十 GB PCIe；没有读者在写之前点名任何 cell，无影响）。
- **反压**：miss 拷贝发生在注意力内核之前、计算流之上，本轮阻塞等待；
  预填充路径用 **STRATA_KV_PREFETCH**（docs/KV_PREFETCH.md）把「下一层的上传」挪到
  独立 stream 与当前层计算重叠，ready/release 事件保序，复用同一 staging 池；
  32K/64K 预填充时间 -5.3%/-4.1%，串行 KV 等待 ~939ms/1553ms 移出关键路径。
- **失败模式**：镜像 pin 不住（内存不够/驱动拒绝）⇒ 打印明确错误：降低 context 或
  去掉 `--kv-resident`（layer.cpp:915）；不做静默降级。
- **可观测**：计数器在 idle+quiet 时读回（kv_stream.cu 头注），有专门测试
  `test_kv_stream_counters.py`。

### 1.4 与"会话续用"机制的配合（同属单会话故事）

- **prompt-cache 周期快照 + 尾部快照**（docs/PROMPT_CACHE_TAIL.md）：长 prompt 分块
  prefill，按间隔保存中间态；分支共享前缀时按最近边界复用，避免重读。
- **消息边界缓存**（docs/MESSAGE_BOUNDARY_CACHE.md，`STRATA_CACHE_MESSAGE_BOUNDARY=1`）：
  在上一轮边界额外保存一个快照，最后一句被改时从边界续读而不是重读全部历史
  （限 ≥8192 token 前缀、尾 ≤1024）。
- 这两个是 prefill 复用（算过的不再算），冷热窗口是 KV 存放（存的不再全放显存），
  正交组合。FusionKV 的对应物是 vLLM 自带 prefix caching + 外部 KV 连接器，不需要重做。

## 2. 机制 B：VMM 弹性 KV 池（`--kv-grow`）

（src/core/layer.cpp:624-720，src/program/generate.cpp:7858、12282，include/strata/core/layer.hpp:285，
docs/VRAM_ELASTIC.md）

- **地址**：每个 mode-0 state 一个 `VmmRange`：`cuMemAddressReserve` 预留**全上下文**
  大小的虚拟地址；每张池数组（K codes/K scales/V…）起始于 chunk 边界（2 MiB 对齐），
  `off[]`/`per_slot[]` 记录数组偏移与每槽字节。
- **物理页**：初始只映射 `g_kv_elastic_init = 16384` cell（layer.cpp:626）；
  `pool_slots_mapped` = 所有数组都映射到的整槽数（取最小，数组各自连续增长）；
  `pool_grow(p, cells, take)` 把缺口 chunk 映射上——chunk 由 `take()` **从专家缓存借**；
  缩小时反向归还。⇒ 显存在「专家缓存 ↔ KV」之间按需流动，两边都是同一 VMM 粒度。
  会话文件续用（conversation_file）加载时也先 grow 到文件 cell 数（generate.cpp:12282）。
- **为什么上层零感知**：VA 预留后**指针永不变**；内核读到的越界地址必然已被 grow 映射
  （grow 由"请求到达的 cell 数"驱动，先于读发生）；新映射 chunk 在映射时清零
  （`qsa_state_zero` 只清已映射部分，未映射的映射时清，layer.cpp:963 区域）。
- **与 CUDA graph**：Strata 主内核本就按 page_table 间接寻址，弹性只动物理页，
  graph capture 不受影响。这是「小显存启动」最干净的一条路：**启动显存 =
  16384 cell + 会话推进中到达过的 cell**，而不是 `--max-context` 全款。
- **对偶的 C**：`--vram-elastic` 把专家缓存换成分段分配，`POST /v1/vram
  {"reserve_mib": N}` 运行时 unmap 尾段让出显存、`null/0` 收回；有下限（128 槽 +
  最小 prompt loan）、部分成功仍回 200（比较 `vram_free_mib` 与 `reserve_mib`）、
  忙时 409。协议细节见 docs/VRAM_ELASTIC.md —— FusionKV 的运行时控制口照此语义设计。

## 3. KV 机制 × 注意力后端的耦合分析（Strata 实测 ↔ vLLM 对照）

### 3.1 Strata：冷热窗口与注意力后端是**内核级共生**（qsa_layer，src/core/layer.cpp:1195-1290）

每 token 的注意力序列（全部在同一计算流上、全部在 graph capture 区内）：

```
kv_append_*_step(pool, page_table, step, kcur, vcur, …, &st.host)   ← 写路径①：量化写显存槽
                                                                   ← 写路径②：同一内核顺手写锁页镜像
indexer_append → block_scores/top-k → b.ids                        ← DSA 稀疏选择（cap 个块/token）
qsa_kv_resolve(st, g, b.ids, …)          ← layer.cpp:1261「选择点名的块先全部驻留」
    └─ resolve_kernel(1 block)：查表解析命中 + 登记缺口 + 时钟扫描选 victim
       copy_kernel(n miss)：pinned 镜像 → 显存槽，16B 向量整块拷
qsa_decode_attn_step(qcur, pools, b.ids, …)  或  kv_gather_*_step → attend
    └─ 只读显存池；page_table 间接寻址；绝不触碰 host 指针
reset_kernel：回合尾清引用位
```

六条耦合（FusionKV 设计的标尺）：

| # | 耦合 | 位置 | 含义 |
|---|---|---|---|
| C1 | 所有 KV 读写必经 `page_table` 间接 | append/gather/fast-attn 内核签名（qsa.cu:595-773） | 换任何注意力后端都必须支持页表间接——这正是 vLLM block_table 的原生形态 |
| C2 | **每步读集 = indexer top-k**，resolve 先于注意力（layer.cpp:1261） | `qsa_kv_resolve(st,…,b.ids,…)` | 需求分页的可行性完全依赖 DSA 稀疏注意力：无 indexer ⇒ 读集=整个前缀 ⇒ 每步整前缀拷回，机制退化。**这是最大的隐性耦合** |
| C3 | 写镜像在 append/gather **内核内部**（`mirror ? &hk : nullptr`，K8V4 分两半各写各的） | layer.cpp:1213-1218 | 权威镜像的内核内双写；换后端=换内核，镜像写点必须随内核移植 |
| C4 | 全链 CUDA-graph 可捕获：步计数在设备（`st.step`）、grid 用容量不用序列长（layer.cpp:1240 注释）、**pinned 镜像地址固定** | `qsa_layer` "EVERY ENTRY POINT FROM HERE ON IS THE CAPTURABLE ONE" | 锁页地址一旦变化（realloc/换页）capture 即失效——cudaHostGetDevicePointer 一次取定 |
| C5 | 量化格式（int8/q4/k8v4）由注意力/ gather 内核**原生读取**；分层搬运是纯 memcpy（同格式） | runs_of + 各内核 | 后端与「池格式」耦合，与「分层机制」不耦合 ⇒ 传输永远字节不变 |
| C6 | 预取重叠在**跨层调度层**（prefill 上传走独立流 + ready/release 事件），不在内核内 | docs/KV_PREFETCH.md | 内核只保序（事件），不感知流 |

### 3.2 vLLM：连接器与注意力后端**完全解耦**（本仓库实测）

```
model_executor/layers/attention/kv_transfer_utils.py:51  connector.wait_for_layer_load(layer_name)  ← 每层注意力前
                                             :57         connector.save_kv_layer(layer_name, kv_cache, …) ← 每层注意力后
vllm/v1/worker/gpu/kv_connector.py:94,143                start_load_kv / wait_for_save（graph 外，eager）
vllm/compilation/breakable_cudagraph.py:86               @eager_break_during_capture 必须外包 @maybe_transfer_kv_layer，
                                                         否则 wait/save 的副作用录进 graph，replay 挂死
vllm/v1/simple_kv_offload/worker.py:288                  store 前在计算流记 compute-done 事件再排 D2H
                                                         （v1 overlapped：KV 正在被本步写，直接拷会读脏，issue #45704）
```

- **C1 对应**：vLLM 各后端（FA/FlashInfer/Triton）内核内已按 `block_table` 间接 ⇒ FusionKV
  把数据拷进块即可，**注意力内核零改动**，无需移植 C3。
- **C2 无对应，必须换粒度**：vLLM 常规后端的读集=整个前缀且内核是外部库，
  **不可能**在注意力内部做 resolve-copy 需求分页。FusionKV 把 Strata 的
  「每步读集驱动」换成「**请求准入驱动**」：外部命中 → 异步整段载入分配的块，
  `load_kv_async=True` 让调度器等拷完再跑（`get_finished` 报告）；写侧走
  `save_kv_layer`/`wait_for_save`（= 镜像写点，块粒度、层间天然重叠）。
  若未来 vLLM 上出现 DSA/indexer 后端，per-layer 需求分页可作为可插拔策略复用同一 worker。
- **C4 对应**：连接器拷贝都在 eager 段发起（start_load_kv/wait_for_save），与 cudagraph 正交；
  per-layer 等待若启用则声明 `requires_piecewise_for_cudagraph=True`（可断裂 graph 保证 hook 在
  eager 段）。默认用「异步载入完成前不调度该请求」策略，逐层等待关闭 ⇒ 任何 graph 模式都安全。
- **弹性池 × 注意力（B）**：块 id 是数据（block_table 内容），kernel 地址 = base + id×stride，
  任何 graph 模式下 id 都不可预知 ⇒ **grow 必须发生在步前的 eager 段**（收到 SchedulerOutput
  后、forward 前，把步内可能触达的最大块 id 区间映射齐）；replay 期间映射集稳定，
  收缩只在引擎 idle 时做。指针恒定 ⇒ capture 的内核参数与 graph 永不失效。
- **C3/C5 对应**：vLLM 若开 fp8 KV，主机镜像存**同 dtype 同布局**（不重量化）⇒ 位一致，
  与 Strata 同格式镜像同理。

### 3.3 由耦合分析落定的 FusionKV 形态

| 决策 | 依据 |
|---|---|
| 分层拷贝走连接器钩子，**不 patch 注意力内核** | C1↔vLLM block_table；C3 在 vLLM 无内核内镜像点，用 save_kv_layer 代替 |
| 默认「准入时整段异步载入」而非逐步需求分页 | C2：vLLM 无稀疏读集 |
| 逐层预取重叠做成 opt-in（per-layer event 组 + `requires_piecewise_for_cudagraph`） | C6↔maybe_transfer_kv_layer + breakable graph 契约 |
| D2H 一律排在计算流 compute-done 事件之后 | worker.py:288 / #45704 |
| 弹性 grow 在步前 eager 段、shrink 只在 idle；镜像/池同格式零重量化 | §3.2 弹性×注意力、C5 |

### 3.4 驻留下界：为什么 vLLM 开了 offload 仍要「一轮完整上下文」，Strata 不必（本树实测）

**vLLM 下界的四条成因（R1–R4，行号实测）**

| # | 成因 | 位置 |
|---|---|---|
| R1 | **准入按整序列计块**：可调度 ⇔ 池能装下整条序列的块；外部命中的 token 也要先从池分配 GPU 块再载 | vllm/v1/core/sched/scheduler.py:3292（`kv_cache_manager.coordinator.get_num_blocks_to_allocate`）；scheduler.py:1247-1432（`allocate_slots(delay_cache_blocks=load_kv_async)`） |
| R2 | **内核契约**：`block_table` 每项=物理 VRAM 块；FA/FlashInfer/Triton/MLA 内核每 token 遍历 `block_table[:ceil(ctx/bs)]` 按 `base+id×stride` 解引用。内核态**没有 miss/resolve 路径**——可能被读的块在 op 启动前必须已映射且已填充 | C1/C2 的另一面；稀疏内核例外见 L3 |
| R3 | **offload 时序=先载后算**：异步载入请求 park 在 `WAITING_FOR_REMOTE_KVS`（scheduler.py:1495），worker `get_finished` 报告完成才进 RUNNING（scheduler.py:3404-3408 的 still-waiting 检查）；块整个载入期被占用。offload 改变字节**来源**（重算 vs 传输），不改变「计算时全驻留」 | 连接器生命周期 |
| R4 | **前缀缓存身份绑定**：block id 即缓存键（hash 挂物理块、`ref_cnt>0` 不可动），驱逐以整块释放为粒度——不存在「缩小正在使用的上下文」 | vllm/v1/core/block_pool.py |

（外加 R5：容量流水线 `determine_available_memory → num_gpu_blocks` 纯按字节计，无 VA≫物理概念。）

⇒ **推论**：vLLM 的 KV 显存下界 = 单最长在算请求完整上下文 × 并发 + 水位；offload 不碰此下界。

**Strata 逃逸机制（M1–M5，对偶）**：M1 权威在主机（pinned 镜像内核内双写，显存池纯 cache、槽位随时换逻辑块，无「块 id=缓存条目」身份，缩小无约束）；M2 每步读集=indexer top-k，注意力前 `qsa_kv_resolve` 点命中块拷进驻留池（layer.cpp:1261）⇒ 驻留下界=top-k 工作集×层×并发，**与上下文长度无关**；M3 resolve/copy/reset 全设备侧、graph 可捕获（页入决策不需 host 调度回合）；M4 prefill 稠密读集靠跨层流式上传与计算重叠（KV_PREFETCH）；M5 准入不看 KV 字节（池按 VA 给出、物理页惰性映射）。**硬前提**：M2 要求每 token 读集 ≪ 上下文——稠密模型每 token 整传不可行，Strata 单会话窗口同样只在 qsa 稀疏选择下成立（C2 耦合，胶水层抬不动）。

**FusionKV 移植分层（L1/L2/L3）**

- **L1**（计划本体=A+B）：分层拷贝 + 弹性 VA。缓存存储字节≈0（镜像权威、命中异步整载），但 R2/R3 未动 ⇒ 在算请求整上下文驻留下界不变。
- **L2**（opt-in）：逐层载入与计算重叠 + 物理页惰性映射。省等待与缓存字节，不动稠密驻留下界。
- **L3（完整移植=真 Strata 模式，仅稀疏后端）**：本树已有完整 DSA/sparse indexer 体系——
  `vllm/model_executor/layers/sparse_attn_indexer.py:1002`（`SparseAttnIndexer` CustomOp，topk 为设备张量）、
  `vllm/v1/attention/backends/mla/indexer.py:266`（`DeepseekV32IndexerBackend`），模型侧
  deepseek_v32/v4/v41、glm5next、step5（`use_dsa_backend`）、qwen4_exp QSA、hy_v4 全在树。移植五件套：
  ① 调度放松：连接器能力位 `supports_partial_residency`——外部命中请求发起载入即可调度，解除
  `WAITING_FOR_REMOTE_KVS` 门控（scheduler.py:1486-1523 准入/状态机 ~50-100 行）；
  ② worker 钩子：indexer op 与 attention op 之间插 resolve（DSA 层 forward 加 hook 点 ~20-30 行或
  backend 子类化）：读 topk 设备张量→FusionKV resolve 内核（查表→缺口→victim→H2D，计算流、设备驱动、
  可入 graph——`qsa_kv_resolve` 同形态）；注意力内核零改动（sparse 内核本就只 gather top-k 块，实现时
  用 memcheck 验证其严格性，若有整表扫描点则该 op 前整体物化）；
  ③ 弹性池（B 已有）⇒ 物理驻留=工作集、水位可观测；
  ④ 权威归属：save_kv_layer D2H 已有，GPU 前缀缓存字节归零语义，命中判定整体委托 FusionKV；
  ⑤ prefill：独立流逐层组上传+事件（移植 KV_PREFETCH 编排），chunk 计算与上传重叠。
  本质是把 vLLM 不变式「scheduled ⇒ 全部块已填充」放松为「稀疏 op 只保证 top-k 填充」。

## 4. FusionKV 映射（python + C++，vLLM 外置模块）

vLLM v1 已有两块天然挂点。分层（A）严格零改动（连接器经
`KVTransferConfig.kv_connector_module_path` 动态加载，本树
`simple_cpu_offload_connector` 为官方参考实现）；弹性池（B）经用户许可（m00210）
允许**适量 vLLM 修改**：优先以最小 in-tree hook 落地（`allocate_kv_cache` 注入
`alloc_fn`、`determine_available_memory` 支持 VA>物理 的报告），FusionKV 同时保留
无 patch 时的运行时 monkey-patch 回退路径。

1. **KVConnector**（`vllm/distributed/kv_transfer/kv_connector/`，经
   `KVTransferConfig.kv_connector_module_path` 动态加载外部模块；本仓库
   `simple_cpu_offload_connector` 是官方 CPU 分层参考）。
   关键回调：`get_num_new_matched_tokens`/`update_state_after_alloc`（调度侧命中与分配）、
   `start_load_kv`/`wait_for_layer_load`（读侧，逐层等待可做到逐层预取重叠）、
   `save_kv_layer`/`wait_for_save`（写侧 D2H）、`register_kv_caches`（拿到
   `dict[str, torch.Tensor]` 真实层缓冲）、`request_finished`（会话结束决定去留）。
2. **KV 缓冲分配点** `vllm/v1/worker/utils.py:564 allocate_kv_cache`（由
   `gpu_model_runner.py:229` 导入、6935 调用；`determine_available_memory` 在
   `gpu_worker.py:641`）。general plugin 入口（`vllm.plugins.load_general_plugins`）
   可在 worker 启动前打运行时 patch，把这块缓冲换成弹性 VMM 张量。

| Strata | FusionKV（机制 A，默认开） | FusionKV（机制 B，`FUSIONKV_ELASTIC=1`） |
|---|---|---|
| 锁页内存全量镜像（恒等布局） | `HostArena`：mmap + cudaHostRegister，block 恒等布局（与 GPU 层缓冲同构，直接按 block 拷贝） | — |
| 热窗口 + clock sweep | vLLM 的 BlockPool LRU 就是热窗口（block 分配即驻留）；FusionKV 只加**host 槽位二级索引**（token prefix hash → host slot，带 second-chance） | — |
| resolve/copy kernel | 不需要逐 cell resolve（vLLM 块粒度）；一个 CUDA 拷贝内核批量 H2D/D2H `runs`，按 `save_kv_layer`/`wait_for_layer_load` 节拍与计算重叠（= KV_PREFETCH 思想） | — |
| `--kv-grow` VMM 池 | — | C++ `VmmRange`（dlopen libcuda，HIP 走 `hsa`/同 API）；`torch.frombuffer` 式张量包装；`grow_to(blocks)` 由 block table 最大 id 驱动（worker 每步 `start_load_kv` 检查）；映射时清零 |
| `POST /v1/vram` | — | 进程内控制：`FusionKVPool.reserve_mib(n)` + unix socket（可选），语义照抄 VRAM_ELASTIC（下限、部分成功、忙时 409） |
| 位一致 parity 测试 | CPU 后端等价性测试（fallback 路径逐位） + GPU 上 streamed vs full 对照 | 映射/读写/清零等价性单测 |

**为什么 A、B 正交且都要**：A 让 KV 总量不受显存约束（超出的在 RAM，命中免重算）；
B 让**启动显存**随实际使用增长（vLLM 默认按 `determine_available_memory` 把 KV 池
一次性 `torch.empty` 满）。B 的指针不变性保证 CUDA graph 与已 capture 的 kernel 参数
在 grow/shrink 全程有效。两者叠加：小显存启动 + 大上下文 + 前缀命中 + 与专家/其它
进程共享显存。

> 完整移植（驻留 < 单轮上下文）见 §3.4-L3：仅稀疏-decode 后端（本树 DSA/QSA 系）可行，
> 作为路线里程碑实现，不影响 A/B 的正交语义。

## 5. 文件行号索引（本仓库实测）

```
src/kernels/cuda/kv_stream.cu:1-286      resolve/copy/reset/ring 内核、runs_of、counters、stage/unstage
src/core/layer.cpp:624                   ElasticPool（--kv-grow），g_kv_elastic_init=16384
src/core/layer.cpp:738-757               kv_plan：mode 0/1/2
src/core/layer.cpp:769-771               qsa_set_kv_resident / min 20480
src/core/layer.cpp:868-884               KvStreamMap 各表分配
src/core/layer.cpp:900-940               锁页镜像分配 + pin 失败提示(915)
src/core/layer.cpp:953                   page table 初始化（恒等 / reset / ring）
src/core/layer.cpp:989                   回合尾：清驻留不清镜像
src/core/layer.cpp:1211                  append 双写主机镜像
src/core/layer.cpp:1261                  注意力前：点名块先驻留
include/strata/core/vmm.hpp               VmmRange（cuMemAddressReserve + chunk handles）
src/program/generate.cpp:489/839/2487    --kv-grow 选项；:7858 弹性分配；:12282 文件续用 grow
docs/VRAM_ELASTIC.md                     POST /v1/vram 协议全文
docs/KV_PREFETCH.md                      STRATA_KV_PREFETCH 逐层上传重叠及实测
docs/PROMPT_CACHE_TAIL.md                尾部快照
docs/MESSAGE_BOUNDARY_CACHE.md           消息边界缓存
src/kernels/kv_stream_parity.cpp         streamed vs resident 位一致 parity 测试
vllm（参考树 /home/amd/dsh-work/vllm）：
  vllm/config/kv_transfer.py:81          kv_connector_module_path
  vllm/distributed/kv_transfer/kv_connector/factory.py:103   module_path 动态注册
  vllm/distributed/kv_transfer/kv_connector/v1/base.py       KVConnectorBase_V1 / SupportsHMA
  vllm/v1/worker/utils.py:564            allocate_kv_cache
  vllm/v1/worker/gpu_model_runner.py:229,6935,7054  导入点/调用点/register_kv_caches 点
  vllm/v1/worker/gpu_worker.py:641       determine_available_memory
```

## 6. FusionKV 实现状态（v1 完成时快照）

模块 ↔ 机制对照（本文档 §1/§2 → 代码）：

| Strata 机制 | FusionKV 实现 | 备注 |
|---|---|---|
| §1 锁页主机镜像 + 双写 | worker.py `_alloc_pinned` + eager 扫描/store（`store_scan_step`）；懒式= lazy_store+high_water | 双写时机改为回合尾批量（vLLM 无 append 钩子），快照在 request_finished 补尾 |
| §1 runs_of 游程合并拷贝 | csrc/copy_engine.cpp + plan.py（纯函数孪生，单测覆盖） | 一次 cudaMemcpyAsync/游程 |
| §1 页表点名驻留（注意力前） | 异步整块 H2D + WAITING_FOR_REMOTE_KVS 门控 | vLLM 调度器已有该状态位，无需逐层钩子 |
| §1 LRU+二级机会页表驱逐 | block_index.py（second-chance、pin、前缀链） | |
| §2 VmmRange 弹性 VA | csrc/vmm.cpp + elastic.py + patch.py 钩子1 | 单一大 backing 分配处拦截（utils.py:592 assert 单 backing） |
| §2 chunk 按需映射、单调增长 | worker._grow_elastic：元数据 scheduled_max_block_id 水位 → ensure_covered | 水位来自 patch.py 钩子2（BlockPool.get_new_blocks） |
| §2 POST /v1/vram 显存弹性 API | 未接（v1 用 FUSIONKV_*  env 静态配置；trim_to 保留在 elastic.py） | 服务器控制面后续接入 |
| §2 广告显存 > 物理 | patch.py 钩子3（determine_available_memory 取 max） | L3 地板移植核心 |
| §3 注意力零耦合（页表间接寻址） | 适配层仅用 KVConnectorBase_V1 公开面 + 3 个运行时 monkey-patch | 无 vLLM 源码修改 |
| §3.3 两阶段释放/抢占冲刷 | scheduler.py 保持协议（hold→注入 finished_sending）；need_flush | 镜像 simple_cpu_offload manager 契约 |

启动路径：`pip install ./fusionkv`（setup.py 编译 csrc → fusionkv._C；缺头文件可 FUSIONKV_PY_INC）→ plugin 入口 `fusionkv.plugin:load`（注册 "FusionKV" connector；FUSIONKV_VA_BYTES 存在时 patch.install()）。

验证状态：无 GPU 环境 16 项单测全绿（调度器契约流程 7 + 块索引 + stub 连接器 4 + plan/链式哈希）；setup.py 构建路径含 TORCH_EXTENSION_NAME=-D 修复已复验。GPU 实机步骤固化为 scripts/gpu_e2e.sh（注册 → 贪心同输出 → 主机命中加速断言 → 弹性超配启动断言），待实机执行。
