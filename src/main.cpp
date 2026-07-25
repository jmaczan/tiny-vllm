#include <iostream>
#include <numeric>
#include <fstream>
#include "cuda_to_hip.h"
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"
#include "request_scheduler.h"

using json = nlohmann::json;

constexpr int MAX_NEW_TOKENS_GENERATED = 20; // TODO: parameterize it with program arguments
constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;
constexpr int N_LAYERS = 16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_K_HEADS = 8;
constexpr int NUM_V_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4;
constexpr int VOCAB_SIZE = 128256;
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>
constexpr int MAX_SEQ_LEN = 2048;            // TODO: make it tunable
constexpr int BATCH_SIZE = 2;                // TODO: not even close to being good, it's just here to have batching
constexpr int MAX_PROMPT_LEN = 512;          // TODO: arbitrary, tunable
constexpr int MAX_BUFFER_SIZE = std::max(MAX_PROMPT_LEN, BATCH_SIZE);
constexpr int BLOCK_SIZE = 16; // TODO: tunable as well, defined the size of a single page in pagedattn
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;                         // * 2 because K and V
constexpr size_t KV_CACHE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024; // TODO: 2GB
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;      // 2048 / 16 = 128
constexpr int NUM_BLOCKS = KV_CACHE_SIZE_BYTES / BLOCK_BYTES;     // 2*1024*1024*1024/(16*512*2*2) = 65536
constexpr int MAX_SEQUENCES = BATCH_SIZE;

int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

struct Weights  // LLaMA的权重
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

//我们先设计一个空的架子，开辟一片内存，但是没有具体数值，将格子大小划分好后，在将model里面的数字给填进来

int loadWeights(Weights &weights)
{
    if (checkGPUStatus() != 0)
    {
        return 1;
    }

    // READ SAFETENSORS
    std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
    if (!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file\n";
        safetensors_file.close();
        return 1;
    }

    // READ SAFETENSORS HEADER SIZE
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
   
    // READ SAFETENSORS HEADER
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);
    // READ OFFSETS OF EVERY LAYER (TENSOR) TO KNOW WHERE EVERY LAYER STARTS AND ENDS IN THE MEMORY
    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offset = 0;
    for (auto &[key, value] : header_json.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }
        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset)
        {
            max_offset = offset_end;
        }
        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
    }

    void *model_weights;
    cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
    safetensors_file.close();
    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    // TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but maybe it would be convenient
    //       to store dimensions somewhere for even easier access?
    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
    for (int i = 0; i < N_LAYERS; ++i)
    {
        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
    }
    return 0;
}

// TODO: clean up this mess lol XD (I mean, the arguments list is so long, but maybe that's unavoidable, I don't know yet)
void prefill(Request &request, std::vector<int> &prompt, int &prompt_len, int prefill_len, std::vector<bool> &is_slot_free, int slot, int *gpu_input_tokens, nv_bfloat16 *input_embeddings, Weights &weights, nv_bfloat16 *hidden_state, nv_bfloat16 *rms_norms, nv_bfloat16 *&q_proj, nv_bfloat16 *buf_2048_1, cublasHandle_t cublas_handle, float &q_proj_alpha, float &q_proj_beta, float &k_proj_alpha, float &k_proj_beta, float &v_proj_alpha, float &v_proj_beta, nv_bfloat16 *prefill_attn_scores, float &attn_alpha, float &attn_beta, nv_bfloat16 *&attn_scores_v, float &attn_scores_v_alpha, float &attn_scores_v_beta, nv_bfloat16 *&o_proj, nv_bfloat16 *buf_2048_2, float &o_proj_alpha, float &o_proj_beta, float &gate_alpha, float &gate_beta, nv_bfloat16 *gate, float &up_alpha, float &up_beta, nv_bfloat16 *up, nv_bfloat16 *&down, float &down_alpha, float &down_beta, float &embed_alpha, float &embed_beta, nv_bfloat16 *embed_proj, std::vector<nv_bfloat16> &embed_proj_cpu, std::vector<std::vector<int>> &generated_tokens, std::vector<int> &last_generated_tokens, std::vector<int> &current_prompt_len, __nv_bfloat16 *k_proj_temp_buf, __nv_bfloat16 *v_proj_temp_buf, std::vector<int> &block_table, int *block_table_gpu, std::vector<int> &free_blocks, std::vector<bool> &block_is_cached, PrefixCacheManager *prefix_cache, __nv_bfloat16 *kv_cache, bool use_prefix_cache)
{
    prompt = request.tokens;
    prompt_len = prefill_len;
    is_slot_free[slot] = false;
    request.slot = slot;

    cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len * sizeof(int), cudaMemcpyHostToDevice);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, prompt_len);

    cudaMemcpy(hidden_state,    //Transformer 的隐藏状态张量
               input_embeddings,
               prompt_len * EMBEDDING_LENGTH * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {

        //计算Q K V矩阵
        rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

        // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
        // it perceives my matrices as transposed
        // there's a trick where C = A * B == C^T = B^T * A^T
        // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
        // so I need to do: Q^T = wq ^T * inputs
        // the beauty is that we don't need to transpose Q^T back to Q
        // because cublas sees the output as column-major
        // so it's in fact transposed
        // final dim (num_tok, EMBEDDING_LENGTH)
        q_proj = buf_2048_1;
        cublasStatus_t q_proj_status = cublasGemmEx(cublas_handle,   //句柄，用于管理 CUDA 上下文
                                                    CUBLAS_OP_T,    //第一个矩阵 A 转置（op(A) = A^T）
                                                    CUBLAS_OP_N,    //第二个矩阵 B 不转置（op(B) = B）
                                                    EMBEDDING_LENGTH,  //输出矩阵 C 的行数
                                                    prompt_len,    //输出矩阵 C 的列数（当前 batch 的 token 数）
                                                    EMBEDDING_LENGTH,   //矩阵乘法的公共维度（A 的列数 = B 的行数）
                                                    &q_proj_alpha,    //alpha 系数，alpha * op(A)*op(B)
                                                    weights.w_q[layer],  //第 layer 层的 Q 投影权重
                                                    CUDA_R_16BF,    //矩阵 A 的数据类型：BF16（bfloat16）
                                                    EMBEDDING_LENGTH, 
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &q_proj_beta,
                                                    q_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
        // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
        // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
        cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_alpha,
                                                    weights.w_k[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_beta,
                                                    k_proj_temp_buf,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // same as K projection
        cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_alpha,
                                                    weights.w_v[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_beta,
                                                    v_proj_temp_buf,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // RoPE now
        // 计算RoPE
        rope(q_proj, prompt_len, EMBEDDING_LENGTH);
        rope(k_proj_temp_buf, prompt_len, KV_DIM);

        //KV CACHE
        // PagedAttention - scatter K and V into blocks
        // slot - index within batch
        // layer - index of layer
        // ceil(prompt_len/BLOCK_SIZE) = number of blocks needed to allocate in block table
        for (int token_idx = 0; token_idx < prompt_len; token_idx += BLOCK_SIZE)
        {
            //处理边界情况，当num_tokens_to_copy大于BLOCK_SIZE时，那这一块就要分块，以BLOCK_SIZE个token作为一块
            int num_tokens_to_copy = prompt_len - token_idx;  
            if (num_tokens_to_copy > BLOCK_SIZE)
            {
                num_tokens_to_copy = BLOCK_SIZE;
            }
            // read index of physical block from logical block_table
            // if -1, then need to allocate the new block
            // pop from free_blocks
            // write its value to block_table on the same position we read from
            // compute address of this block table in kv_cache
            // write tokens to it
            int block_idx = token_idx / BLOCK_SIZE;    //计算逻辑块号

            //接下来从页表中找到对应的物理块号，如果没有就分配一个新的物理块号
            //slot:当前序列在batch中的槽位索引 N_LAYERS:Transformer层数  MAX_BLOCKS_PER_SEQ:每个序列每层最多的块数  layer:当前 Transformer 层索引
            int block = block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx];
            size_t prefix_len = token_idx + num_tokens_to_copy;
            int cached_block = -1;
            if (use_prefix_cache) {
                cached_block = prefix_cache->lookupPrefix(prompt, prefix_len, layer);
            }
            if (block == -1)
            {
                if (use_prefix_cache && cached_block >= 0)
                {
                    block = cached_block;
                    block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx] = block;
                    block_is_cached[block] = true;
                }
                else
                {
                    int physical_block_idx = free_blocks.back();
                    free_blocks.pop_back();
                    block = physical_block_idx;
                    block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx] = block;
                    if (use_prefix_cache) {
                        int inserted_block = prefix_cache->lookupOrInsertPrefix(prompt, prefix_len, layer, block);
                        if (inserted_block >= 0)
                        {
                            block_is_cached[inserted_block] = true;
                        }
                    }
                }
            }
            else
            {
                // Block already exists for this slot. It was filled by a previous prefill chunk.
            }

            if (!block_is_cached[block])
            {
                // store K
                __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES);
                __nv_bfloat16 *k_proj_ptr = k_proj_temp_buf + token_idx * KV_DIM;
                cudaMemcpy(k_cache_ptr, k_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                // store V
                __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET);
                __nv_bfloat16 *v_proj_ptr = v_proj_temp_buf + token_idx * KV_DIM;
                cudaMemcpy(v_cache_ptr, v_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            }
        }

        // attention scores
        // per head, 64 elements each
        // so total 32 heads
        // Q (num_tok, 2048)
        // K (num_tok, 512)
        // GQA grouping reuses 1 K head per 4 consecutive Q heads
        // Q_head (num_tok, 64)
        // K_head (num_tok, 64)
        // attn_score_head = Q_head * K_head^T / sqrt(64)
        // so: head output dims (num_tok, num_tok)
        // total output (32, num_tok, num_tok)
        for (int i = 0; i < NUM_Q_HEADS; ++i)  //遍历所有的Q头 以头为单位计算注意力分数
        {
            int k_head_idx = i / GQA_Q_TO_K_RATIO;
            __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
            __nv_bfloat16 *k_head = k_proj_temp_buf + k_head_idx * HEAD_DIM;
            __nv_bfloat16 *attn_score_head = prefill_attn_scores + prompt_len * prompt_len * i; // 计算每个

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            prompt_len,
                                                            prompt_len,
                                                            HEAD_DIM,
                                                            &attn_alpha,
                                                            k_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,
                                                            q_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,
                                                            &attn_beta,
                                                            attn_score_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        causalMask(prefill_attn_scores, prompt_len);

        softmax(prefill_attn_scores, prompt_len);

        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
        attn_scores_v = buf_2048_1;
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
            // i * prompt_under_prefill.size() * prompt_under_prefill.size(),  because attn scores is (32, num_tok, num_tok)
            __nv_bfloat16 *attn_scores_head = prefill_attn_scores + i * prompt_len * prompt_len;
            __nv_bfloat16 *v_head = v_proj_temp_buf + v_head_idx * HEAD_DIM;
            __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            HEAD_DIM,
                                                            prompt_len,
                                                            prompt_len,
                                                            &attn_scores_v_alpha,
                                                            v_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,
                                                            attn_scores_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,
                                                            &attn_scores_v_beta,
                                                            output_attn_scores_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        //多头注意力的输出是多个头的简单拼接，需要通过可学习的线性变换（w_o）来整合信息，产生更有意义的特征表示
        //输出投影是多头注意力机制的必要组成部分，用于整合多头信息！
        // output projection, it will be an input for MLP blocks
        // attn_scores_v * w_o^T
        // (num_tok, 2048) * (2048, 2048) -> (num_tok, 2048)
        // same as Q projection, so copy paste
        //
        o_proj = buf_2048_2;
        cublasStatus_t o_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_alpha,
                                                    weights.w_o[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    attn_scores_v,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_beta,
                                                    o_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, o_proj, prompt_len);
        // post attention RMS Norm
        rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], prompt_len);

        //实现SwiGLU前，先要有两个投影：gate投影和up投影
        //接下来这第一个是gate投影
        // SwiGLU time - just MLP + SiLU
        // gate = hidden_state (rms-normed) * mlp_gate_proj ^ T
        // HIDDEN_DIM = 8192
        // (num_tok, 2048) * (2048, 8192) -> (num_tok, 8192)
        // my data is row major so transpose trick
        // gate ^T = (mlp_gate_proj ^ T)^T * hidden_state^T
        // gate ^T = mlp_gate_proj * hidden_state^T
        // (num_tok, 8192)^T = (8192, 2048) * (2048, num_tok)
        // but data is perceived as column major so I need to transpose mlp_gate_proj
        // to make it work
        // m 8192 n num_tok k 2048 lda 2048 ldb 2048 ldc 8192
        cublasStatus_t gate_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  HIDDEN_DIM,
                                                  prompt_len,
                                                  EMBEDDING_LENGTH,
                                                  &gate_alpha,
                                                  weights.mlp_gate_proj[layer],
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  &gate_beta,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // up, the same dims as gate
        // 接下来这第二个是up投影
        cublasStatus_t up_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                HIDDEN_DIM,
                                                prompt_len,
                                                EMBEDDING_LENGTH,
                                                &up_alpha,
                                                weights.mlp_up_proj[layer],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &up_beta,
                                                up,
                                                CUDA_R_16BF,
                                                HIDDEN_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

        // SiLU
        // after_silu = SiLU(gate) * up (element-wise multication)
        // after_silu = gate * (1 / (1 + e^(-gate))) * up
        // gate is dim (num_tok, 8192), up too
        silu(gate, up, prompt_len); // gate = after_silu now

        // 接下来这第三个是down投影
        // down projection
        // output = post-silu * down_proj^T
        // dims: (num_tok, 8192) * (2048, 8192) ^ T = (num_tok, 8192) * (8192, 2048) = (num_tok, 2048)
        // output^T = (down_proj^T)^T * post-silu^T
        // output^T = down_proj * post-silu^T
        // cublas sees them already as transposed so only down_proj I need to transpose
        // dims = (2048, 8192) * (8192, num_tok) = (2048, num_tok)
        // m: 2048 n: num_tok, k: 8192
        // lda: 8192, ldb: 8192, ldc: 2048
        down = buf_2048_2;
        cublasStatus_t down_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  EMBEDDING_LENGTH,
                                                  prompt_len,
                                                  HIDDEN_DIM,
                                                  &down_alpha,
                                                  weights.mlp_down_proj[layer],
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  &down_beta,
                                                  down,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, down, prompt_len);
    }
    //transformer的最后一层输出，还要进行一次RMS Norm
    rmsNorm(hidden_state, rms_norms, weights.norm, prompt_len);

    // 接下来是词表投影， 将语义表示（隐藏状态）转换为词表上的概率分布
    // logits = rms_norms * weights.embed_tokens^T
    // dim rms_norms: (num_tok, 2048), dim embed_tokens: (128256, 2048)
    // logits dim = (num_tok, 2048) * (2048, 128256) = (num_tok, 128256) => m = num_tok, n = 128256, k = 2048
    // I leave this comment above because it shows a bug in my thinking
    // because I use the cublas trick, logits are transposed so m and n should be swapped
    // so m 128256, n num_tok
    // data is row major so we treat it as transposed and use the trick
    // logits^T = ((weights.embed_tokens^T)^T * rms_norms^T
    // logits^T = weights.embed_tokens * rms_norms^T
    // so we need to transpose embed_tokens, because rms_norms already
    // appears to cublas as transposed
    // lda = 2048, ldb = 2048, ldc = 128256

    cublasStatus_t embed_status = cublasGemmEx(cublas_handle,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               VOCAB_SIZE,
                                               prompt_len,
                                               EMBEDDING_LENGTH,
                                               &embed_alpha,
                                               weights.embed_tokens,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               rms_norms,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               &embed_beta,
                                               embed_proj,
                                               CUDA_R_16BF,
                                               VOCAB_SIZE,
                                               CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT);


    //将词表投影结果复制到CPU上，即GPU->CPU
    cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * prompt_len * VOCAB_SIZE, cudaMemcpyDeviceToHost);
    // argmax to get the output token
    // TODO: write a proper kernel for it
    // for now just a simple CPU function
    int last_token_offset = (prompt_len - 1) * VOCAB_SIZE;   //定位最后一个token的logits，推理时我们只关心模型接下来会生成什么，也就是最后一个输入 token 之后的输出。
    //接下来是找到最大值的token索引，也就是模型预测的下一个token
    float max_token = (float)embed_proj_cpu[last_token_offset];
    int max_token_idx = 0;
    for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
    {
        if ((float)embed_proj_cpu[token_idx + last_token_offset] > max_token)
        {
            max_token = embed_proj_cpu[token_idx + last_token_offset];
            max_token_idx = token_idx;
        }
    }
    std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;

    generated_tokens[slot].push_back(max_token_idx);  //将模型预测的下一个token添加到历史生成的token序列中
    last_generated_tokens[slot] = max_token_idx;   //快速获取每个序列的最新 token，供后续 decode 阶段使用
    current_prompt_len[slot] = prompt_len;  //更新当前序列的 prompt 长度，用于后续的 decode 阶段

    // synchronize state of block_table with block_table_gpu
    // TODO: do it more clever and not copy full table unnecessarily
    cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);   //将页表数据从CPU同步到GPU，确保在后续的 decode 阶段中，GPU 能够正确访问到每个序列的 K/V 缓存块信息
}

int main(int argc, char *argv[])
{
    // 创建cuBLAS句柄，作用：初始化 cuBLAS 库，用于执行 GPU 上的矩阵乘法
    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }
    //加载模型权重
    Weights weights{};
    if (loadWeights(weights) != 0)
    {
        return 1;
    }

    // allocator for pagedattn
    // 分配 K/V 缓存内存，用于存储每个序列的 K/V 缓存块
    __nv_bfloat16 *kv_cache;
    cudaMalloc(&kv_cache, KV_CACHE_SIZE_BYTES);
    std::vector<int> free_blocks(NUM_BLOCKS);
    std::iota(free_blocks.begin(), free_blocks.end(), 0);
    std::vector<int> block_table(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);
    std::vector<bool> block_is_cached(NUM_BLOCKS, false);
    int *block_table_gpu;
    cudaMalloc(&block_table_gpu, MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int));   //这是GPU端的页表，用于记录每个序列在每一层使用了哪些物理块

    PrefixCacheManager prefix_cache(128);
    RequestScheduler scheduler(&prefix_cache, BLOCK_SIZE);
    std::queue<Request> request_queue;

    //准备了四个问题，也就是四个不同的 prompt，分别是：
    // PROMPT 0 (What is 2+2?) - length 17
    request_queue.push(Request{0, {128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271}});

    // PROMPT 1 (Name a color.) - length 14
    request_queue.push(Request{1, {128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009, 128006, 78191, 128007, 271}});

    // PROMPT 2 (Say hello.) - length 13
    request_queue.push(Request{2, {128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009, 128006, 78191, 128007, 271}});

    // PROMPT 3 (Capital of France?) - length 14
    request_queue.push(Request{3, {128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009, 128006, 78191, 128007, 271}});

    while (!request_queue.empty()) {
        scheduler.addRequest(request_queue.front());
        request_queue.pop();
    }

    std::vector<int> slot_request_id(BATCH_SIZE, -1);

    // BATCH
    //这个用来记录槽位是否空闲   BATCH_SIZE是批处理的大小，也就是同时处理的序列数量
    std::vector<bool> is_slot_free(BATCH_SIZE, true); // set to false when slot taken, set to true when free

    std::vector<std::vector<int>> generated_tokens(BATCH_SIZE); //内层用来记录第slot个序列生成的所有token，外层则是槽位
    std::vector<int> last_generated_tokens(BATCH_SIZE);  //用来快速获取每个序列的最新 token，供后续 decode 阶段使用
    std::vector<int> current_prompt_len(BATCH_SIZE, 0);  //用来记录当前序列的 prompt 长度，用于后续的 decode 阶段

    // needed to provide contiguous data for decode
    // CPU 端活跃槽位和 token。在 decode 阶段，只处理活跃的序列，跳过已完成的
    std::vector<int> active_slots;
    std::vector<int> active_tokens; 

    //GPU 端活跃槽位和序列长度 供 GPU kernel（如 pagedAttention）使用
    int *gpu_active_slots;
    cudaMalloc(&gpu_active_slots, BATCH_SIZE * sizeof(int));
    int *gpu_seq_lens;
    cudaMalloc(&gpu_seq_lens, BATCH_SIZE * sizeof(int));

    // TODO: recalculate input_tokens_size and prompt_lengths always when there is a change to prompt_under_prefill
    // TODO: right now I handle input manually, it's the least interesting part, will come back to it when continuous batching and pagedattn works

    //CPU 端 prompt
    std::vector<int> prompt;
    int prompt_len;
    //GPU 端输入 token
    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
    //GPU 端输入 token 的嵌入表示  词嵌入向量的维度是 EMBEDDING_LENGTH：2048
    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
   //GPU 隐藏状态区
    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    //GPU 端 RMS 归一化区
    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    // buf_2048_1：Q 投影和注意力输出共享的临时缓冲区
    __nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
    cudaMalloc(&buf_2048_1, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    //q_proj：指向该缓冲区，存储 Q 投影结果
    __nv_bfloat16 *q_proj;
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;

    // K and V cache
    __nv_bfloat16 *k_proj_temp_buf;
    cudaMalloc(&k_proj_temp_buf, MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16));

    __nv_bfloat16 *v_proj_temp_buf;
    cudaMalloc(&v_proj_temp_buf, MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16));

    //K/V 投影 GEMM 的 alpha/beta
    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;

    __nv_bfloat16 *prefill_attn_scores;
    cudaMalloc(&prefill_attn_scores, MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;

    __nv_bfloat16 *attn_scores_v;
    float attn_scores_v_alpha = 1.0f;
    float attn_scores_v_beta = 0.0f;

    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    cudaMalloc(&buf_2048_2, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *o_proj;
    float o_proj_alpha = 1.0f;
    float o_proj_beta = 0.0f;

    __nv_bfloat16 *gate;
    cudaMalloc(&gate, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;

    __nv_bfloat16 *up;
    cudaMalloc(&up, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float up_alpha = 1.0f;
    float up_beta = 0.0f;

    __nv_bfloat16 *down;
    float down_alpha = 1.0f;
    float down_beta = 0.0f;

    //词表投影相关
    __nv_bfloat16 *embed_proj;   //GPU 上的词表投影结果（logits）
    cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * MAX_BUFFER_SIZE * VOCAB_SIZE);
    float embed_alpha = 1.0f;
    float embed_beta = 0.0f;

    std::vector<__nv_bfloat16> embed_proj_cpu;
    embed_proj_cpu.resize(MAX_BUFFER_SIZE * VOCAB_SIZE);   //CPU 上的副本，用于 ArgMax
    
    
    // decode-only allocation  
    //最后生成的 token（GPU）
    int *gpu_last_tokens;
    cudaMalloc(&gpu_last_tokens, BATCH_SIZE * sizeof(int));   
    // TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves

    // K/V 批量缓冲区
    // reused temporary buffers for K and V cache computation during decode
    __nv_bfloat16 *k_proj_batched_buffer;
    cudaMalloc(&k_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);

    __nv_bfloat16 *v_proj_batched_buffer;
    cudaMalloc(&v_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);

    // CUDA Graph support
    cudaStream_t graphStream = 0;
    cudaStreamCreate(&graphStream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool graph_instantiated = false;
    size_t graph_num_active_slots = 0;
    // runtime toggle: set env USE_CUDA_GRAPH=0 to disable graph capture/launch
    bool use_cuda_graph = true;
    const char* use_graph_env = getenv("USE_CUDA_GRAPH");
    if (use_graph_env != nullptr) {
        use_cuda_graph = (atoi(use_graph_env) != 0);
    }
    // runtime toggle for chunked prefill: set CHUNKED_PREFILL=0 to disable chunking
    bool chunked_prefill = true;
    const char* chunked_env = getenv("CHUNKED_PREFILL");
    if (chunked_env != nullptr) {
        chunked_prefill = (atoi(chunked_env) != 0);
    }
    // runtime toggle for prefix cache / APC: set USE_PREFIX_CACHE=0 to disable prefix caching
    bool use_prefix_cache = true;
    const char* use_prefix_cache_env = getenv("USE_PREFIX_CACHE");
    if (use_prefix_cache_env != nullptr) {
        use_prefix_cache = (atoi(use_prefix_cache_env) != 0);
    }

    //遍历所有的槽位，对等待请求执行prefill，将结果存储到KV缓存中
    for (int slot = 0; slot < is_slot_free.size(); ++slot)
    {
        if (!is_slot_free[slot])
        {
            continue; // slot taken, skip
        }
        Request* req = scheduler.startNextWaitingRequest(slot);
        if (!req)
        {
            break;
        }
        slot_request_id[slot] = req->request_id;
        generated_tokens[slot].clear();

        size_t remain = req->tokens.size() - req->prefill_cursor;
        int chunk_len = static_cast<int>(std::min(remain, chunked_prefill ? (size_t)BLOCK_SIZE : remain));
        int prefill_len = static_cast<int>(req->prefill_cursor + chunk_len);
        prefill(*req, prompt, prompt_len, prefill_len, is_slot_free, slot, gpu_input_tokens, input_embeddings, weights, hidden_state, rms_norms, q_proj, buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha, embed_beta, embed_proj, embed_proj_cpu, generated_tokens, last_generated_tokens, current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table, block_table_gpu, free_blocks, block_is_cached, &prefix_cache, kv_cache, use_prefix_cache);
        req->prefill_cursor += chunk_len;
        current_prompt_len[slot] = prompt_len;
        if (req->prefill_cursor >= req->tokens.size()) {
            req->state = RequestState::Decoding;
            if (!prompt.empty()) {
                last_generated_tokens[slot] = prompt.back();
            }
        }
    }

    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is EMBEDDING_LENGTH length bf16 vector
    // retrieved from model weights based on token's value

    // PREFILL

    // DECODE
    // since now I operate always on index 0 for all values and for current_position_token for new K and V

    while (true) // exit condition irrelevant for now, since it's an inference server that's supposed to run foreveeer!!!
    {
        // 清空所有活跃槽位的生成 token
        active_slots.clear();
        active_tokens.clear();
        //decode中的这一块prefill是用于  在decode阶段中，检查是否有槽位刚刚被释放（序列完成）
        //如果有新prompt，分配给空闲槽位    
//         比喻：
//         餐厅有 4 张桌子（BATCH_SIZE = 4）

// 初始状态：
// ┌──────┬──────┬──────┬──────┐
// │ 桌1  │ 桌2  │ 桌3  │ 桌4  │  ← 所有桌子都是空的
// └──────┴──────┴──────┴──────┘

// 第一批顾客来了：
// ┌──────┬──────┬──────┬──────┐
// │ 顾客A│ 顾客B│ 顾客C│ 顾客D│  ← 4 个顾客入座
// └──────┴──────┴──────┴──────┘

// 过了一会儿：
// ┌──────┬──────┬──────┬──────┐
// │ 顾客A│ 顾客B│      │ 顾客D│  ← 顾客C吃完走了，桌3空了
// └──────┴──────┴──────┴──────┘

// 新顾客来了：
// ┌──────┬──────┬──────┬──────┐
// │ 顾客A│ 顾客B│ 顾客E│ 顾客D│  ← 新顾客E坐到桌3
        for (int slot = 0; slot < BATCH_SIZE; ++slot)
        {
            if (is_slot_free[slot])
            {
                if (scheduler.waitingCount() == 0)
                {
                    continue;
                }
                Request* req = scheduler.startNextWaitingRequest(slot);
                if (req)
                {
                    slot_request_id[slot] = req->request_id;
                    generated_tokens[slot].clear();
                    size_t remain = req->tokens.size() - req->prefill_cursor;
                    int chunk_len = static_cast<int>(std::min(remain, chunked_prefill ? (size_t)BLOCK_SIZE : remain));
                    int prefill_len = static_cast<int>(req->prefill_cursor + chunk_len);
                    prefill(*req, prompt, prompt_len, prefill_len, is_slot_free, slot, gpu_input_tokens, input_embeddings, weights, hidden_state, rms_norms, q_proj, buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha, embed_beta, embed_proj, embed_proj_cpu, generated_tokens, last_generated_tokens, current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table, block_table_gpu, free_blocks, block_is_cached, &prefix_cache, kv_cache, use_prefix_cache);
                    req->prefill_cursor += chunk_len;
                    current_prompt_len[slot] = prompt_len;
                    if (req->prefill_cursor >= req->tokens.size()) {
                        req->state = RequestState::Decoding;
                        last_generated_tokens[slot] = prompt.back();
                    }
                }
            }
            else
            {
                Request* req = scheduler.getRequestById(slot_request_id[slot]);
                if (req && req->state == RequestState::Prefill)
                {
                    size_t remain = req->tokens.size() - req->prefill_cursor;
                    int chunk_len = static_cast<int>(std::min(remain, chunked_prefill ? (size_t)BLOCK_SIZE : remain));
                    int prefill_len = static_cast<int>(req->prefill_cursor + chunk_len);
                    prefill(*req, prompt, prompt_len, prefill_len, is_slot_free, slot, gpu_input_tokens, input_embeddings, weights, hidden_state, rms_norms, q_proj, buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha, embed_beta, embed_proj, embed_proj_cpu, generated_tokens, last_generated_tokens, current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table, block_table_gpu, free_blocks, block_is_cached, &prefix_cache, kv_cache, use_prefix_cache);
                    req->prefill_cursor += chunk_len;
                    current_prompt_len[slot] = prompt_len;
                    if (req->prefill_cursor >= req->tokens.size()) {
                        req->state = RequestState::Decoding;
                        last_generated_tokens[slot] = prompt.back();
                    }
                }
            }

            Request* req = scheduler.getRequestById(slot_request_id[slot]);
            if (!is_slot_free[slot] && req && req->state == RequestState::Decoding) {
                active_slots.push_back(slot);
                active_tokens.push_back(last_generated_tokens[slot]);
            }
        }




        int num_active_slots = active_slots.size();    //active表示当前正在运行的，num_active_slots表示当前正在运行的槽位数量
        if (num_active_slots == 0)
        {
            if (scheduler.waitingCount() == 0)
            {
                break; // 没有活跃序列且没有新请求，退出
            }
            continue;  // 还有新请求，继续循环
        }

        // copy useful data to gpu  。将当前正在运行的槽位的生成 token、槽位索引   供 pagedAttention 使用，确定哪些槽位需要处理
        cudaMemcpy(gpu_last_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_active_slots, active_slots.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> seq_lens(num_active_slots);  //存储每个正在运行的token当前长度
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            seq_lens[slot] = current_prompt_len[active_slot] + 1;
        }
        cudaMemcpy(gpu_seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice);

        //将每个任务的最新的token转化为词嵌入，供后续计算使用
        bool use_graph = (use_cuda_graph && graph_instantiated && num_active_slots == graph_num_active_slots);
        bool do_capture = (use_cuda_graph && !graph_instantiated && num_active_slots > 0);

        if (use_graph) {
            // launch cached graph for this shape
            cudaGraphLaunch(graphExec, graphStream);
            cudaStreamSynchronize(graphStream);
        } else {
            // execute (and possibly capture) the per-layer decode path
            if (do_capture) {
                cudaStreamBeginCapture(graphStream, cudaStreamCaptureModeGlobal);
                cublasSetStream(cublas_handle, graphStream);
            }

            embeddingGatherDecode(gpu_last_tokens, num_active_slots, hidden_state, weights.embed_tokens, do_capture ? graphStream : 0);

            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], num_active_slots);
            q_proj = buf_2048_1;
            // q proj (num_prompts, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &q_proj_alpha,
                         weights.w_q[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda
                         rms_norms,        // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb
                         &q_proj_beta,
                         q_proj, // C
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldc
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // k proj (1, 512), writing output to next position in current layer's K cache
            // K proj = rms_norms (num_prompt, 2048) * W_k (512, 2048)
            // W_k is actually stored as 512, 2048 (out features, in features)
            // so that's why we need to transpose it
            // all the data is stored in row major and cublas reads it as column major
            // so all the data appears as transposed
            // so data actually apppears as (2048, num_prompt) * (2048, 512)
            // the output of matmul will also be produced as transposed, so we can say that
            // in our mental model we talk about K_proj^T
            // and to get K_proj^T we can do transposition trick and write the cublas call as
            // W_k^T * rms_nroms
            // so we end up with: K_proj^T = W_k^T (512, 2048) * rms_norms (2048, num_prompt)
            // result dim is K_proj^T = (512, num_prompt)
            // but it's transposed, so in fact we get correct output dimension (num_prompt, 512)
            // it was great for num_prompt=1, but the problem is that prompts have different length
            // that's why we have vector of current_prompt_len, but also we can't write to K_proj
            // directly, so I write to temp buffer kv_proj_batched_buffer and the scatter
            // output to K_proj in a loop
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,           // m = 512
                         num_active_slots, // n = num prompts
                         EMBEDDING_LENGTH, // k = 2048
                         &k_proj_alpha,
                         weights.w_k[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
                         // so the gap between subsequent elements is 2048
                         rms_norms, // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb, same reason for rms_norms
                         &k_proj_beta,
                         k_proj_batched_buffer, // TODO C
                         CUDA_R_16BF,
                         KV_DIM, // ldc = 512
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // same
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_active_slots,
                         EMBEDDING_LENGTH,
                         &v_proj_alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &v_proj_beta,
                         v_proj_batched_buffer,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                ropeDecode(&q_proj[slot * EMBEDDING_LENGTH], current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(k_proj_batched_buffer + slot * KV_DIM, current_prompt_len[active_slot], KV_DIM);
            }

            // PagedAttn scatter k and v from a temp buffer, like in the prefill  将新计算的K/V写入分页缓存，供后续注意力计算使用
            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                int seq_len = current_prompt_len[active_slot]; // + generated tokens?
                int logical_block_idx = seq_len / BLOCK_SIZE;
                int token_in_block_idx = seq_len % BLOCK_SIZE;
                int block = block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
                if (token_in_block_idx == 0)
                {
                    int physical_block_idx = free_blocks.back();
                    free_blocks.pop_back();
                    block = physical_block_idx;
                    block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
                }
                __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *k_proj_ptr = k_proj_batched_buffer + slot * KV_DIM;
                if (do_capture) cudaMemcpyAsync(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, graphStream);
                else cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *v_proj_ptr = v_proj_batched_buffer + slot * KV_DIM;
                if (do_capture) cudaMemcpyAsync(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, graphStream);
                else cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            }

            // synchronize block table on cpu with block table on gpu (for attention)
            if (do_capture) cudaMemcpyAsync(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice, graphStream);
            else cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

            pagedAttention(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, buf_2048_1, do_capture ? graphStream : 0);

            o_proj = buf_2048_2;
            // (1, 2048) * (2048, 2048) -> (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &o_proj_alpha,
                         weights.w_o[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf_2048_1,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &o_proj_beta,
                         o_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, o_proj, num_active_slots);

            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], num_active_slots);

            // MLP
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &gate_alpha,
                         weights.mlp_gate_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &gate_beta,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (1, 2048) * (2048, 8192) -> (1, 8192)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &up_alpha,
                         weights.mlp_up_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &up_beta,
                         up,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            silu(gate, up, num_active_slots);

            down = buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         HIDDEN_DIM,       // k
                         &down_alpha,
                         weights.mlp_down_proj[layer],
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         &down_beta,
                         down,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, down, num_active_slots);
            }

            // if we were capturing, finish capture and instantiate graph
            if (do_capture) {
                cudaGraph_t tmp;
                cudaStreamEndCapture(graphStream, &tmp);
                cudaGraphInstantiate(&graphExec, tmp, NULL, NULL, 0);
                graph = tmp;
                graph_instantiated = true;
                graph_num_active_slots = num_active_slots;
                // restore cublas to default stream
                cublasSetStream(cublas_handle, 0);
            }
        }

        rmsNorm(hidden_state, rms_norms, weights.norm, num_active_slots);

        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,       // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &embed_alpha,
                     weights.embed_tokens,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &embed_beta,
                     embed_proj,
                     CUDA_R_16BF,
                     VOCAB_SIZE,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        //先将词表投影结果复制到CPU上，即GPU->CPU 然后在 CPU 上找最大概率的 token。             
        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * num_active_slots * VOCAB_SIZE, cudaMemcpyDeviceToHost);

        float max_token = 0.0;
        int max_token_idx = 0;
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            max_token = (float)embed_proj_cpu[slot * VOCAB_SIZE]; // TODO: verify if float is good enough in place of nvbf16
            max_token_idx = 0;
            for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
            {
                if ((float)embed_proj_cpu[slot * VOCAB_SIZE + token_idx] > max_token)
                {
                    max_token = embed_proj_cpu[slot * VOCAB_SIZE + token_idx];
                    max_token_idx = token_idx;
                }
            }
            // TODO: wrap with #ifdef DEBUG
            std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
            
            //判断是否是结束符或者当前序列长度是否达到最大长度，如果是，则释放槽位和缓存块
            if (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID || current_prompt_len[active_slot] == MAX_SEQ_LEN - 1)
            {
                is_slot_free[active_slot] = true;
                if (slot_request_id[active_slot] != -1) {
                    scheduler.finalizeRunningRequest(slot_request_id[active_slot]);
                    slot_request_id[active_slot] = -1;
                }
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    for (int logical_block_idx = 0; logical_block_idx < MAX_BLOCKS_PER_SEQ; ++logical_block_idx)
                    {
                        int block_idx = active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
                        if (block_table[block_idx] != -1)
                        {
                            int block_id = block_table[block_idx];
                            if (!block_is_cached[block_id])
                            {
                                free_blocks.push_back(block_id);
                            }
                            block_table[block_idx] = -1;
                        }
                    }
                }
                cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);
            }
            else
            {
                last_generated_tokens[active_slot] = max_token_idx;
                generated_tokens[active_slot].push_back(max_token_idx);
                current_prompt_len[active_slot] = current_prompt_len[active_slot] + 1;
            }
        }
    }
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}
