# tiny-vllm (fork: Continuous Batching + APC + Chunked Prefill + CUDA Graph)

这是基于 jmaczan/tiny-vllm 的一个 fork，主要增加和集成了以下工程与性能特性：

- 自动前缀缓存（Automatic Prefix Cache, APC）：使用 `PrefixCacheManager` 在预填充（prefill）阶段按 prefix 长度缓存并重用 KV blocks，从而减少重复的 K/V 写入与拷贝。
- Chunked Prefill：将较长的 prompt 拆分为 block/chunk（以 `BLOCK_SIZE` 为单位）进行分段 prefill，以控制内存/带宽峰值并使 APC 更高效。
- Continuous Batching（持续批处理）：保留原仓库的连续批处理设计并在此基础上优化了槽位/页表逻辑以配合 chunked prefill 与 APC。
- CUDA Graph 支持：添加运行时开关以在合适场景下捕获并重放 CUDA Graph（减少内核/调度开销），运行时通过 `USE_CUDA_GRAPH` 环境变量控制。

本仓库的设计目标仍然是以最少的代码实现高性能 LLM 推理（Llama 3.2 1B 作为参考），但在工程化与吞吐/延迟折衷上做了针对性的优化，便于做基准测试与对比分析。

---

## 主要文件（高层）

- `src/main.cpp`：主流程，包含 prefill/ decode、批处理调度与环境开关（`USE_PREFIX_CACHE`、`CHUNKED_PREFILL`、`USE_CUDA_GRAPH`）。
- `src/prefix_cache.h` / `src/prefix_cache.cpp`：APC 的核心实现，基于前缀哈希映射到物理 block id。
- `src/kernels.cu` / `src/kernels.cuh`：CUDA kernel（embedding gather、RMSNorm、RoPE、softmax、residual 等）。
- `include/json.hpp`：单文件 JSON 解析依赖（nlohmann/json）。

---

## 构建（示例）

确保已准备 `model.safetensors` 在仓库根目录并已安装 CUDA 与必要工具。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

或者使用仓库自带脚本：

```bash
./build.sh
```

---

## 运行与环境变量

运行时可以通过环境变量控制三项重要功能：

- `USE_PREFIX_CACHE`：是否启用自动前缀缓存（APC）。0=禁用，1=启用（默认）。
- `CHUNKED_PREFILL`：是否按块拆分 prefill。0=禁用，1=启用（默认）。
- `USE_CUDA_GRAPH`：是否捕获/重放 CUDA Graph。0=禁用，1=启用（默认开启，但仅在支持且已实例化时生效）。

示例：比较 APC 开关的基准（将输出及 time 信息重定向）：

```bash
# 基线：禁用 prefix cache
/usr/bin/time -v env USE_PREFIX_CACHE=0 CHUNKED_PREFILL=1 USE_CUDA_GRAPH=1 ./build/tiny-vllm > /tmp/tiny_vllm_prefix_cache_off.log 2>&1

# 启用 APC
/usr/bin/time -v env USE_PREFIX_CACHE=1 CHUNKED_PREFILL=1 USE_CUDA_GRAPH=1 ./build/tiny-vllm > /tmp/tiny_vllm_prefix_cache_on.log 2>&1
```

记录 `Elapsed (wall clock) time`、`Max resident set size`、以及自定义日志（输出 token、token index、KV cache hit/miss 相关信息）可作为性能/正确性比较的依据。

---

## 实现细节速览

- APC（`PrefixCacheManager`）使用 FNV-1a 风格的哈希结合 layer id 与 prefix 长度计算 key，map 到物理 block id。lookup 与 lookupOrInsert 两个接口分别用于只查找或查找并插入映射。
- Chunked prefill 在 `prefill()` 调用处按 `BLOCK_SIZE` 切分 prompt（见 `src/main.cpp`），每个 chunk 会单独进行 embedding/gemm/KV 写入与页表更新。对已经命中的 prefix，直接用 cache 中的 block 跳过写入，减少 Device->Device memcpy。
- CUDA Graph：在主循环中判断 `graph_instantiated` 与当前活动槽位数，当满足条件时捕获图并复用以减少多次 kernel launch 的开销。通过 `USE_CUDA_GRAPH` 环境变量运行时开关。

---

## 基准与验证建议

- 对比项：APC on/off、Chunked prefill on/off、CUDA Graph on/off、不同 batch/slot 数量、不同 prompt 长度分布。
- 指标：吞吐 (tokens/sec)、平均与 P95 延迟、GPU 内存占用、KV-cache hit rate、以及生成 correctness（token index/样本输出）。
- 建议先在小样本上验证输出正确性（检查 `Output token:` 日志，确认 token index 非全 0），再做大规模吞吐测试。

---

## 在简历里如何写（示例）

下面给出若干可直接拷贝到简历的中英文条目，第一行为简短一句话概述，后面为可展开的要点（项目经理 / 负责人角度）：

中文（推荐写法 - 项目负责人 / 个人项目）：

- 项目：tiny-vllm（fork） — 高性能 LLM 推理引擎（C++/CUDA）；新增自动前缀缓存（APC）、Chunked Prefill、持续批处理与 CUDA Graph 优化
  - 设计并实现 `PrefixCacheManager` 自动前缀缓存，按 prefix 长度重用 KV blocks，显著减少重复写入与内存带宽占用。
  - 将 prefill 按 `BLOCK_SIZE` 分块（Chunked Prefill），与 APC 协同工作以降低内存峰值与 I/O，提升系统稳定性。
  - 集成并测试 CUDA Graph 捕获/重放，减少 kernel 调度开销；加入运行时开关以便回归测试与对比实验。
  - 负责端到端基准设计与实验（脚本化 run、/usr/bin/time 日志、吞吐/延迟对比），用实验数据验证优化效果并产出可复现结果。
  - 技术栈：C++17、CUDA（cuBLAS）、CMake、safetensors、nlohmann/json；在 Linux + NVIDIA GPU 环境下开发与验证。

英文（简短版本，适合简历）：

- tiny-vllm (fork) — High-performance LLM inference engine (C++ / CUDA)
  - Implemented Automatic Prefix Cache (APC) to reuse KV blocks by prefix, reducing redundant K/V writes and GPU memory bandwidth.
  - Introduced Chunked Prefill and integrated with continuous batching to lower memory peaks and improve throughput/latency trade-offs.
  - Added CUDA Graph capture/replay path with runtime toggle, enabling lower kernel launch overhead in steady-state workloads.
  - Ran systematic benchmarks, produced reproducible logs and analysis comparing APC on/off and graph on/off configurations.

提示：如果你在基准中测得了明确的提升（如吞吐或延迟的百分比），把具体数字写进简历（例如“通过 APC 将平均延迟降低 X% / 吞吐提升 Y%”），这会更有说服力。

---

如果你希望，我可以：

1. 把这个 README 写回仓库根目录的 `README.md`（覆盖）或创建 release-ready 的 `README.md`；
2. 根据你手头的 benchmark 日志自动生成一段“结果摘要”（含表格或对比图表）；
3. 帮你把项目写成 GitHub 仓库简介（用于简历链接页），并提供一段短描述用于社交平台/领英。 

请选择下一步或告诉我你希望 README 的语言偏好（中文/英文或双语），以及是否要我直接覆盖仓库中的 `README.md`。 
