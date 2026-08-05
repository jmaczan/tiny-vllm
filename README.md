# tiny-vllm-extend

这是一个基于 `tiny-vllm` 的 CUDA/C++ 推理扩展工程，聚焦于长前缀与长 token 工作负载下的 benchmark 表现。这个 fork 的目标不仅是补齐运行时统计能力，更是让执行路径更可观测、更易复用，并更适合前缀缓存感知的长输入场景。

## 项目概览

本仓库保持实现结构相对紧凑与可调试，同时加入了统一的性能统计路径，以及针对长公共前缀 prompt 的前缀缓存感知执行流。

当前工作区中保留了两条可直接对照的执行路径：

- `src/`：拓展版实现，包含真实的 prefix-cache 统计、运行时指标、调度器级控制流和 benchmark 报告。
- `src_origin/`：基线参考路径，用于与拓展版在同一 workload 下做直接对照。

## 这个 fork 相比上游基线新增了什么

相较于原始单路径参考实现，本 fork 引入了多项适合长上下文 benchmark 的运行时与调度能力：

- PrefixCacheManager
  - 实现真实的 prefix 哈希复用与 block 级 lookup / insertion 统计。
  - 产生可量化的 KV-cache 命中率，而不是占位式或 dummy 输出。

- Chunked Prefill
  - 将长 prompt 按 block 大小拆分为若干 chunk 逐步推进。
  - 降低一次性 prefill 的峰值压力，使长前缀执行更稳定。

- Request scheduler 与 running / waiting / finished 状态机
  - 维护请求生命周期，并更适合连续批处理与长输入场景下的推进逻辑。

- 运行时性能统计
  - 输出 prefill 耗时、峰值显存、总运行时长、端到端吞吐以及 TPOT 等关键指标。

- 运行时开关
  - `USE_PREFIX_CACHE`
  - `CHUNKED_PREFILL`
  - `USE_CUDA_GRAPH`
  - `MAX_NEW_TOKENS_GENERATED`

## Benchmark 环境

当前工作区中完成验证所使用的环境是：

- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- 驱动版本：`535.247.01`
- 显存：`6144 MiB`
- CUDA 工具链：`CUDA 11.3`（`nvcc`）
- 代码中对应的模型配置：`Llama-3.2-1B-Instruct`
- 当前 benchmark 使用的实现常量：
  - `BATCH_SIZE = 2`
  - `MAX_PROMPT_LEN = 2048`
  - `MAX_SEQ_LEN = 8192`
  - `BLOCK_SIZE = 128`
  - `N_LAYERS = 16`

## 已验证的长前缀比较结果

以下对比结果是在同一条长前缀请求轨迹、同一长 token workload 下验证得到的。

| 场景 | 构建模式 | 总生成 token 数 | 总运行时间 | 吞吐 | TPOT（end-to-end） | TPOT（decode-only） | KV-cache 命中率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 补丁版（`src`） | Release | 15358 | 4604.86 ms | 3335.17 tokens/s | 0.299835 ms/token | 0.0841213 ms/token | 66.6667% |
| 原版（`src_origin`） | `nvcc -O2` | 15358 | 16242.3 ms | 945.557 tokens/s | 1.05758 ms/token | 0.836126 ms/token | N/A |

### 结论

- 在同一条长前缀输入下，补丁版表现出明显的吞吐优势。
- 相比原版，补丁版在该 benchmark 下的吞吐大约提升了 `3.53x`。
- 补丁版的端到端 TPOT 也从约 `1.06 ms/token` 降到了约 `0.30 ms/token`。

## 为什么这个 fork 有意义

这个仓库的目标是回答一个很具体的问题：

> 在长前缀工作负载下，新增的运行时统计能力与前缀缓存感知调度路径，是否能够转化为相对于原始实现的真实性能提升？

从已验证的 benchmark 结果看，答案是肯定的。

这个 fork 特别适用于以下场景：

- 长上下文 benchmark 实验
- prefix 复用分析
- prefill / decode 性能剖析
- 在统一请求轨迹下进行端到端推理速度测量

## 项目结构

- `src/main.cpp`：拓展版运行主循环、指标统计、调度器流转和运行时开关
- `src/request_scheduler.h` / `src/request_scheduler.cpp`：请求生命周期与批处理控制
- `src/prefix_cache.h` / `src/prefix_cache.cpp`：prefix 哈希复用与缓存统计
- `src/kernels.cu` / `src/kernels.cuh`：CUDA kernel 路径
- `src_origin/main.cpp`：用于直接对照的基线实现

## 构建与运行

### 1）构建拓展版

```bash
cd ./tiny-vllm-extend
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build-release
cmake --build build-release -j4
```

### 2）构建原版基线

```bash
cd ./tiny-vllm-extend
mkdir -p build-origin
/usr/local/cuda-11.3/bin/nvcc -std=c++17 -O2 -I src_origin -I include -I src -g -DDEBUG \
  -o build-origin/tiny-vllm-origin src_origin/main.cpp src_origin/kernels.cu -lcublas -lcudart
```

### 3）运行拓展版

```bash
cd ./tiny-vllm-extend
./build-release/tiny-vllm
```

### 4）运行原版对照

```bash
cd ./tiny-vllm-extend
./build-origin/tiny-vllm-origin
```

## 运行时开关

拓展版执行流支持以下运行时控制参数：

- `USE_PREFIX_CACHE=1`：启用 prefix-cache 复用
- `USE_PREFIX_CACHE=0`：关闭 prefix-cache 复用
- `CHUNKED_PREFILL=1`：启用 chunked prefill
- `CHUNKED_PREFILL=0`：关闭 chunked prefill
- `USE_CUDA_GRAPH=1`：启用 graph capture 复用
- `USE_CUDA_GRAPH=0`：关闭 graph capture 复用
- `MAX_NEW_TOKENS_GENERATED=0`：允许 benchmark 在不受人工 token 上限硬截断影响下运行

示例：

```bash
cd ./tiny-vllm-extend
USE_PREFIX_CACHE=1 CHUNKED_PREFILL=1 USE_CUDA_GRAPH=1 MAX_NEW_TOKENS_GENERATED=0 ./build-release/tiny-vllm
```

## 实践性结论

这个项目本质上是一个紧凑型工程 benchmark harness，同时也是一个面向长前缀推理评估的优化 fork。它的核心贡献在于四个方面：

1. 可量化的运行时统计能力；
2. 前缀缓存感知的复用机制；
3. chunked prefill 与调度器协同的执行流；
4. 与原始实现的同 workload、同输入轨迹对照能力。

从 benchmark 证据看，拓展版不仅报告了更完整的性能数据，而且在同一长上下文负载下实现了明显的 runtime 提升。

