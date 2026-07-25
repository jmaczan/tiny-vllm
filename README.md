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

