#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bf16.h>
#define __nv_bfloat16 __hip_bfloat16
#define nv_bfloat16   __hip_bfloat16
#else
#include <cuda_bf16.h>
#endif

// prefill
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens);
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, nv_bfloat16 *norm_weights, int num_tokens);
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim);
void causalMask(__nv_bfloat16 *input, int num_tokens);
void softmax(__nv_bfloat16 *input, int num_tokens);
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens);
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens);

// decode
void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens);
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim);
void softmaxDecode(__nv_bfloat16 *input, int seq_len);

// pagedattn
void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output);