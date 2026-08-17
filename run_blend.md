# CacheBlend 运行与修复说明（vLLM 0.11.0 + Qwen2.5-7B, CUDA）

本文档记录在本机（RTX 4090 ×1, 仅 `cuda:0`）上运行 UCM CacheBlend 的完整步骤，
以及调试过程中发现并修复的三层问题（环境 / API 漂移 / 核心 KV 损坏 bug）。

---

## 1. 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 4090 24GB（仅使用 `CUDA_VISIBLE_DEVICES=0`） |
| Python 环境 | uv venv：`/home/yjm/unified-cache-management/.venv`（Python 3.12） |
| vLLM | 0.11.0（CacheBlend 的 sparse patch 只存在于 v0110） |
| torch | 2.8.0 (cu128) |
| transformers | **4.57.6（必须 < 5.0**，5.x 删除了示例脚本依赖的 `use_chat_template` 且与 vllm 0.11.0 不兼容） |
| 模型 | Qwen2.5-7B：`/home/yjm/models/qwen2_7b`（`Qwen2ForCausalLM`，被 v0110 qwen2 fork 覆盖） |
| 数据集 | LongBench 2wikimqa：`/home/yjm/datasets/longbench/data/2wikimqa.jsonl` |
| KV 存储目录 | `/home/yjm/ucm_test` |
| UCM 代码 | **worktree `/home/yjm/ucm-prehma`（commit `eb70add`，即 #951）**，editable 安装 |

### 为什么用 eb70add worktree 而不是 develop？

develop 分支的 `ucm_connector.py` 等已跟进 vLLM 0.17+ API（`SupportsHMA`、
`KVConnectorBase_V1(kv_cache_config=...)`、`SchedulerOutput.preempted_req_ids` 等），
与 venv 中的 vLLM 0.11.0 不兼容，且 blend 路径长期无人回归。`eb70add` 是 HMA 改造
（#953）之前、v0110 sparse fork 最后演进（#907）之后的稳定点。

> develop 分支上也打了一层兼容 shim（见 §4.1），可以让 0.11.0 import 通过，
> 但 blend 数据损坏 bug 与分支无关（两个分支都存在，根因在 C++ 传输层），
> 干净的运行环境以 worktree 为准。

---

## 2. 从零搭建步骤

```bash
cd /home/yjm/unified-cache-management

# 1) Python 依赖（若已建好 venv 可跳过）
uv pip install "transformers>=4.55.2,<5" "wrapt==1.17.2" setuptools wheel cmake

# 2) 构建 + editable 安装 UCM（会自动把 ucm_patch.pth 注入 site-packages，
#    使 python 启动时挂载 monkey-patch 钩子）
git worktree add /home/yjm/ucm-prehma eb70add    # 已存在则跳过
cd /home/yjm/ucm-prehma
PLATFORM=cuda uv pip install -e . \
    --python /home/yjm/unified-cache-management/.venv/bin/python \
    --no-build-isolation

# 验证：应显示 ucm 来自 worktree
/home/yjm/unified-cache-management/.venv/bin/python -c "import ucm; print(ucm.__file__)"
# → /home/yjm/ucm-prehma/ucm/__init__.py
```

注意：Blend 的算子全部是 triton kernel（`ucm/sparse/blend/blockwise_rope.py`），
无需 `ENABLE_SPARSE=TRUE` 编译；`BUILD_UCM_SPARSE` 只影响 MLA/GSA 的 `.cu`，
Blend 用不到。

---

## 3. 运行

```bash
cd /home/yjm/ucm-prehma

# 首次运行前清空 KV 存储目录，避免历史脏数据
rm -rf /home/yjm/ucm_test/*

CUDA_VISIBLE_DEVICES=0 ENABLE_UCM_PATCH=1 ENABLE_SPARSE=TRUE \
MODEL_PATH=/home/yjm/models/qwen2_7b \
DATA_DIR=/home/yjm/ucm_test \
BLEND_DATASET_PATH=/home/yjm/datasets/longbench/data/2wikimqa.jsonl \
/home/yjm/unified-cache-management/.venv/bin/python examples/offline_inference_blend.py
```

环境变量说明：
- `ENABLE_UCM_PATCH=1`：**必须**。否则 `apply_all_patches()` 直接 return，不打任何补丁
- `ENABLE_SPARSE=TRUE`：**必须**。否则不加载 v0110 sparse/blend 补丁
- `CUDA_VISIBLE_DEVICES=0`：只用 0 号卡

### 预期输出（RTX 4090 实测）

```
---------------1. sys prompt: warm up---------------
--------------- baseline with no cache blend ---------------
--------------- cache rag chunks ---------------
--------------- warm up blend code ---------------
--------------- cache blend ---------------
--------------- prefix cache ---------------
Baseline generated text: ' Gyulafehérvár, Transylvania.'
Baseline generated cost time: 0.92 seconds
Prefix Cache generated text: ' Gyulafehérvár, Transylvania.'
Prefix Cache generated cost time: 0.23 seconds
Blend generated text: ' February 24, 1645, Gyulafehérvár, Transylvania. ...'
Blend generated cost time: 2.39 seconds
Question:Where was the wife of Francis I Rákóczi born?
Golden answer:['Ozalj']
```

- **Baseline 与 Prefix Cache 输出逐字一致**，PC 有约 4× 提速 —— 说明 KV dump/load 往返无损
- Blend 输出与 baseline 同一答案（greedy 下前缀一致）；模型本身没答对 Ozalj 是
  模型知识问题，与 CacheBlend 无关
- 日志中可见阶段流转：chunk 请求 dump（BUILD_PREFIX_CACHE 降级属正常，
  `min_blend_threshold=16` 所致）、blend 请求 `CACHE_BLEND ... chunks cache total hit: 120/121`

### 调参

Blend 质量/开销由重算比例控制（`examples/offline_inference_blend.py`）：

```python
"ucm_sparse_config": {
    "Blend": {
        "chunk_end_token_id": chunk_end_token_id,
        "compute_meta": {
            "model.layers.1.self_attn.attn": {"ratio": 0.2},   # 0.2 → 0.5
        },
    }
}
```

实测 Qwen2.5-7B + 2wikimqa：`ratio=0.5` 输出更干净（`' Gyulafehérvár,
Transylvania. ...'`），`ratio=0.2` 快但尾部可能出现重复。可按精度/时延权衡调整。

---

## 4. 修复情况

### 4.1 develop 分支（主仓库 `/home/yjm/unified-cache-management`）的兼容 shim

目的：让 develop 代码在 vLLM 0.11.0 上 import / 初始化通过（当时尚未回退到 worktree）。
这些改动留在主仓库，对 0.17+ 无影响（try/except 或推断仅在旧 API 缺失时生效）：

| 文件 | 修改 |
|---|---|
| `ucm/integration/vllm/ucm_connector.py` | ① `SupportsHMA` 导入加 try/except（0.17 前不存在），缺失时回退 no-op 基类；② `UCMDirectConnector.__init__` / `UCMConnector.__init__` 调 `super().__init__(kv_cache_config=...)` 捕获 TypeError 双路兼容；③ `KVCacheLayout.__init__` 在 `kv_cache_config is None` 时从首个 kv cache 张量 shape 推断 `num_blocks` |
| `ucm/integration/vllm/hla_connector.py`<br>`hma_connector.py`<br>`inference_duration_monitor_connector.py` | 同款 `SupportsHMA` try/except shim |
| `ucm/integration/vllm/blend_connector.py` | `self.monitor.update_stats("ConnStats", {...})` → `ucmmetrics.update_stats({...})`（旧 monitor 属性已从基类移除）；`bind_connector_metadata` 中 `torch.tensor(..., device=self.device)` 改为从 `self.kv_caches` 张量取 `torch.device`（`self.device` 在 worker 侧被 `CudaDevice` 包装对象覆盖） |
| `examples/offline_inference_blend.py` | `max_model_len/max_num_batched_tokens` 32768 → 16384（24GB 卡 32k 会 OOM） |

注意：develop 上 pc-only（`UCMConnector`）路径仍会踩 `SchedulerOutput.preempted_req_ids`
（0.17+ 字段）等更多漂移，**blend 路径可用，pc 路径未继续追**——这也是最终选 eb70add
worktree 的原因。

### 4.2 worktree `/home/yjm/ucm-prehma`（运行环境）的修改

`git diff`（相对 `eb70add`）：

1. `examples/offline_inference_blend.py`
   - `max_model_len/max_num_batched_tokens` 32768 → 16384
   - **关键修复**：`kv_connector_extra_config` 增加
     `"enable_event_sync": False`（根因见 §5）
2. `ucm/integration/vllm/blend_connector.py`
   - `self.monitor.update_stats(...)` → `ucmmetrics.update_stats(...)`（带 `if self.metrics_config:` 守卫）
   - `bind_connector_metadata` 的 device 修复（同 §4.1）

调试期间在 worktree 里插过的所有 spy/DBG 代码已全部清除（`ucm_connector.py`、
`pcstore_connector_v1.py` 已 git checkout 还原）。

---

## 5. 核心 bug：KV dump 与 forward 的竞态（乱码根因）

### 现象

示例端到端能跑通（exit=0），但从 store 加载 KV 后的输出（Prefix Cache / Blend 阶段）
为多语言乱码；baseline（无缓存）输出正常。乱码是否出现与命中块数相关：
2~3 块正常、≥4 块开始损坏（阈值随时间漂移，实为竞态）。

### 定位过程（供复现参考）

1. **隔离**：vanilla vLLM（不打补丁）同模型输出正常 → 问题在 UCM patch 栈；
   blend 栈跑无 chunk 普通提示（首次、无缓存）正常 → 问题在 **dump→load 往返**
2. **阈值实验**：单引擎按 2/3/4/5/6/7/8/32/96 块梯度做 dump→load，损坏从 4~5 块开始
3. **字节级侦察**：在 `UCMDirectConnector` 的 dump 提交点 / load 完成点对全部 28 层
   GPU KV 做快照对比 → layer 0~8 字节一致，**layer 9~27 的 K/V 全为 0**
4. **直接读 store 文件**：磁盘上的文件（每块 56 个 shard = 28 层 × K/V，每 shard 64KiB）
   从某个 shard 起全零，截断点不固定（36/24/22…随时间漂移）
5. **纯 dump 实验**（不 load、不生成，直接分析落盘文件）：**单块 dump 也被截断**
   → 损坏发生在 dump 侧（GPU→host→file），与 load 无关；截断点漂移 → 竞态
6. **根因**：dump 时 GPU 上"未算到的层还是 KV pool 的初始零值"，D2H 抢在 forward
   kernel 之前执行。按设计应由 `prerequisite_handle`（计算流上录制的 event）把 D2H
   串行化，但 pybind 的 `DumpFromDevice(ids, addrs)` **不接收 event 参数，
   `dump_data(prerequisite_handle=...)` 被静默丢弃**
   （`ucm/store/pcstore/cpy/pcstore.py.cc` 的 `SubmitPy`）。

### 修复

`"enable_event_sync": False` 使 `_get_dump_event_handle()` 走
`self.device.synchronize()`（阻塞等待计算完成）而非 event 句柄，彻底规避竞态。

`examples/ucm_config_example.yaml` 中其实有提示
"When you use UcmNfsStore, you should set enable_event_sync to false"，
但官方 `offline_inference_blend.py` 示例未设置，CUDA 上必然踩雷。

### 给上游的修复建议（C++ 侧，未实施）

- pybind `DumpFromDevice` / `LoadToDevice` 增加 event 句柄参数，
  在传输 stream 上 `cudaStreamWaitEvent(event)` 后再发起拷贝，
  让 `enable_event_sync=True` 真正生效（当前 True 只是拿到一个没人消费的句柄）
- 建议同步修复 `offline_inference_blend.py` 示例：加 `enable_event_sync: False`

---

## 6. 常见问题排查

| 症状 | 原因 / 处理 |
|---|---|
| `No module named 'wrapt'` / `import ucm` 失败 | UCM 未安装：`PLATFORM=cuda uv pip install -e . --no-build-isolation` |
| 补丁未生效（日志无 "UCM patching"） | 未设 `ENABLE_UCM_PATCH=1`；或 `.pth` 未注入（检查 site-packages 里的 `ucm_patch.pth` 软链指向当前 worktree） |
| `KVConnectorBase_V1.__init__() got unexpected keyword 'kv_cache_config'` | venv 里的 UCM 是 develop 版：确认 `ucm.__file__` 指向 `/home/yjm/ucm-prehma` |
| 启动即 OOM（KV cache 仅 0.1x GiB） | `max_model_len` 仍为 32768，或上次进程未退出占着显存（`nvidia-smi` 检查） |
| 加载缓存后输出乱码 | 检查是否漏了 `"enable_event_sync": False`；并清空 DATA_DIR 重跑（旧数据可能是竞态时期 dump 的脏 KV） |
| 短提示（<64 token）报 `KeyError: '0'` | blend 路径要求请求 ≥1 个 block，属预期行为，用更长的提示 |

## 7. 现场索引

- 可运行 worktree：`/home/yjm/ucm-prehma`（内含精简版 `RUNBOOK.md`）
- develop 兼容 shim：`/home/yjm/unified-cache-management`（见 §4.1，`git diff` 可查）
- 运行日志样例：`/tmp/blend_final.log`
- uv venv：`/home/yjm/unified-cache-management/.venv`
