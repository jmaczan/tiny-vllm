#include "cuda_to_hip.h"
#include "kernels.cuh"
#include <iostream>
#include <vector>

// TODO perhaps share these between main.cpp and kernels.cu to not duplicate them?

constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr float SQRT_HEAD_DIM = 8;
constexpr int NUM_Q_HEADS = 32;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int MAX_SEQ_LEN = 2048; // TODO: make it tunable
constexpr int BLOCK_SIZE = 16;    // TODO: tunable as well, defined the size of a single page in pagedattn
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;                    // * 2 because K and V
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE; // 2048 / 16 = 128


float *d_inv_freq = nullptr; 
float *d_cos_table = nullptr; // [max_seq_len, head_dim]
float *d_sin_table = nullptr; // [max_seq_len, head_dim]

// prefill / shared

// gpu_input_tokens - N tokens
// gpu_input_embeds - N * sizeof(__nv_bfloat16) * 2048
// embed_tokens - (100000+smth, 2048)
// num_input_tokens - N (just N, not N tokens)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{ 
    int workIndex = threadIdx.x + blockIdx.x * 2048;     //全局索引 每个token对应一个block，token展开成向量便是2048维 每个block能运行1024个线程  每个线程处理2个数字
    if (workIndex < num_input_tokens * 2048)    //防止越界
    {
        gpu_input_embeds[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
        gpu_input_embeds[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{    //Host端来进行调用
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_tokens * 2048)
    {
        rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];   //每个线程，负责计算两个标量的平方，比如threadIdx.x = 0，workIndex = 0，那么rms_vector[0] = input[0] * input[0] + input[1024] * input[1024]的平方
        __syncthreads();
        // tree reduction
        for (int i = 1; i < 1024; i = i * 2)
        {
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];   //树形规约，input[0]+input[1024]+input[1]+input[1025]+...+input[1023]+input[2047]，最终rms_vector[0]就是所有元素的平方和
            }
            __syncthreads();   //确保所有线程完成写入后再进入下一轮
        }
        if (threadIdx.x == 0)
        {
            rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);     //计算平方根，防止除0
        }
        __syncthreads();
        // <(^-^)>
        //每个线程读取 rms_vector[0]，将原始输入除以 RMS 值，再乘以对应的权重 norm_weights，最后转回 bfloat16 写入输出
        output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
        output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
    }
}

// (N, 2048) -> (N, 2048)
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

void init_rope_frequencies(int head_dim, int max_seq_len, float rope_theta,
                           float factor, float low_freq_factor,
                           float high_freq_factor, int original_max_len)
{
        // 2* 表示两个连续的线程为一组，每个线程刚好负责一对相邻的元素，完美覆盖整个向量    proj_dim 是向量的维度，比如2048
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        //这里计算了旋转的基准频率 θ。RoPE 的特点是不同维度的旋转频率不同（指数衰减），这样能捕捉不同距离的相对位置关系。
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = blockIdx.x * theta;
        
        
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];    // 相当于 x
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];   // 相当于 y
        input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));    // x' = x*cos - y*sin
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));  // y' = x*sin + y*cos
    }

    cudaMalloc(&d_cos_table, max_seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_sin_table, max_seq_len * head_dim * sizeof(float));
    cudaMemcpy(d_cos_table, cos_table.data(),
               max_seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin_table, sin_table.data(),
               max_seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
}

void free_rope_frequencies(void)
{
    if (d_inv_freq)
    {
        cudaFree(d_inv_freq);
        d_inv_freq = nullptr;
    }
    if (d_cos_table)
    {
        cudaFree(d_cos_table);
        d_cos_table = nullptr;
    }
    if (d_sin_table)
    {
        cudaFree(d_sin_table);
        d_sin_table = nullptr;
    }
}

__global__ void ropeKernel_llama3(__nv_bfloat16 *input, int num_tokens, int proj_dim,
                                  int head_dim, const float *cos_table, const float *sin_table)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int half_proj = proj_dim / 2;
    int half_dim = head_dim / 2;

    if (tid >= half_proj)
        return;

    int head_idx = tid / half_dim;
    int pair_idx = tid % half_dim;

    int base = token_idx * proj_dim + head_idx * head_dim;
    int idx1 = base + pair_idx;
    int idx2 = base + pair_idx + half_dim;

    float x1 = (float)input[idx1];
    float x2 = (float)input[idx2];

    int table_idx = token_idx * head_dim + pair_idx * 2;
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    input[idx1] = (__nv_bfloat16)(x1 * c - x2 * s);
    input[idx2] = (__nv_bfloat16)(x1 * s + x2 * c);
}

void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel_llama3<<<num_tokens, num_threads>>>(
        input, num_tokens, proj_dim, HEAD_DIM, d_cos_table, d_sin_table);

#ifdef DEBUG
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error)
                  << " (code: " << error << ")" << std::endl;
    }
#endif
}


__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;   //越界处理
    }

    int column = threadIdx.x;
    int row = blockIdx.x % num_tokens;
    if (column > row)   //列大于行，上三角部分，这个地方应当被mask掉，设置为负无穷（-HUGE_VALF）
    {
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability

    //每个线程从全局显存中读取一个属于自己的元素（比如线程 0 读 x0），转换成 float 后，存入共享内存数组 row 中
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;     
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2)   //树形规约求最大值
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < num_tokens; i = i * 2)   //树形规约
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, num_tok, num_tok)
void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// in-place, overwriting a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}

// decode
__global__ void embeddingGatherKernelDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{   //这个 Kernel 的作用就是：“把模型刚刚吐出来的那个词（或这几个词）的 ID，去词表里查出对应的向量，作为下一步计算的输入
    int input_token = gpu_last_tokens[blockIdx.x];   //获取真实的 Token ID
    int workIndex = blockIdx.x * 2048 + threadIdx.x;
    if (workIndex < num_tokens * 2048)
    {
        //拿到 input_token 后，乘以 2048 得到该词向量在巨大词表矩阵中的起始地址。因为单 Block 只有 1024 个线程，所以每个线程依然负责搬运 2 个 bfloat16 数据
        output[workIndex] = embed_tokens[input_token * 2048 + threadIdx.x];
        output[workIndex + 1024] = embed_tokens[input_token * 2048 + threadIdx.x + 1024];
    }
}

#include <cuda_runtime.h>

void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens, cudaStream_t stream)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernelDecode<<<num_tokens, 1024, 0, stream>>>(gpu_last_tokens, num_tokens, output, embed_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void ropeKernelDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim)
{
    if (2 * threadIdx.x + 1 < proj_dim) // TODO: check correctness
    {   
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = position_in_sequence * theta;

        
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1];
        input[2 * threadIdx.x] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernelDecode<<<1, num_threads>>>(input, position_in_sequence, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// seq_len increases by 1 with every new token
__global__ void softmaxKernelDecode(__nv_bfloat16 *input, int seq_len)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    int workIndex = blockIdx.x * MAX_SEQ_LEN + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, seq_len)
void softmaxDecode(__nv_bfloat16 *input, int seq_len)
{
    if (seq_len > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernelDecode<<<NUM_Q_HEADS, seq_len>>>(input, seq_len);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// inside a single particular thread that processes a single position of particular Q head for a particular sequence, for particular layer
__global__ void pagedAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output)
{
    __shared__ float dot_products[2];    //在共享内存中开辟两个 float 空间 用于线程块内的归约（Reduction）。因为 HEAD_DIM 可能是 128 或更大，一个 Warp（32线程）算不完，需要把不同 Warp 算出的部分和汇总到这里
    int active_slot = blockIdx.x; // active_slot == seq_id
    int slot = gpu_active_slots[active_slot];
    int q_head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;   //判断该头用第几组KV Cache
    __nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];  //加载 Q：每个线程从全局显存中读取 一个 bfloat16 类型的 Query 元素，并缓存在寄存器中。这是当前线程负责计算的那个维度的值

    int seq_len = gpu_seq_lens[active_slot];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // for online softmax https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    float current_max = -INFINITY;
    float acc = 0.0f;
    float d = 0.0f; // denominator, same name as in paper above

    //Page Attention循环，这是代码最复杂的部分，实现了 FlashAttention 的 Tiling 逻辑
    for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx)
    {
        // 1. 物理地址转换 (PagedAttention 核心)
        int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
         // 2. 计算块内有效 Token 数（处理最后一个块不满的情况）
        int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
        for (int token = 0; token < tokens_in_block; ++token)
        {
            // 3. 复杂的指针运算加载 K 和 V

            __nv_bfloat16 *k = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            __nv_bfloat16 *v = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + V_OFFSET + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            // 代码通过指针运算，直接定位到 kv_cache 中对应物理块、对应 Token、对应 Head、对应维度的地址
            float qk = (float)q * (float)*k;   // 4. 计算 Q · K (点积的一部分)
            // tree reduction within current warp, thread 0 gets sum of all 32 elements within warp
            // could be done with __syncthreads but accessing memory of other threads in warp is op、

            //刚刚只是每个线程计算了一项，这里通过 __shfl_down_sync 指令，让 Warp 内的 32 个线程把各自算的乘积加起来
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 16);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 8);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 4);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 2);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 1);
            // 执行完后，Warp 内所有线程的 qk 寄存器里都存着完整的点积和

            //接下来进行汇总：假设 HEAD_DIM 是 64，那么会有 2 个 Warp（线程 0-31 和 32-63） 
            //线程 0 和 32 分别把自己 Warp 算出的和存入共享内存
            //线程 0 将两者相加，并除以缩放因子，得到最终的 Attention Score（Logit）
            if (thread_id == 0)
            {
                dot_products[0] = qk;
            }
            if (thread_id == 32)
            {
                dot_products[1] = qk;
            }
            __syncthreads();
            if (thread_id == 0)
            {
                dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
            }
            __syncthreads();
            float dot_product = dot_products[0];



            // online softmax  
            float new_max = current_max;
            if (dot_product > current_max)
            {
                new_max = dot_product;
            }
            float correction_factor = expf(current_max - new_max);
            current_max = new_max;
            float exp_score = expf(dot_product - current_max);
            d = d * correction_factor + exp_score;
            acc = acc * correction_factor + exp_score * (float)*v;
        }
    }
    output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;  
}

void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output, cudaStream_t stream)
{
    pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM, 0, stream>>>(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, output);
}