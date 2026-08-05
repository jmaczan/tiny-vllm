#pragma once

// CUDA-to-HIP compatibility header for tiny-vllm
// Keeps CUDA spellings in source and aliases them to HIP on AMD GPUs

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hipblas/hipblas.h>

// bfloat16 type mappings
#define __nv_bfloat16 __hip_bfloat16
#define nv_bfloat16   __hip_bfloat16

// CUDA runtime -> HIP runtime
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaGetLastError        hipGetLastError
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMemGetInfo          hipMemGetInfo
#define cudaDeviceProp          hipDeviceProp_t
#define cudaError               hipError_t
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess

// cuBLAS -> hipBLAS
#define cublasHandle_t          hipblasHandle_t
#define cublasStatus_t          hipblasStatus_t
#define cublasCreate            hipblasCreate
#define cublasDestroy           hipblasDestroy
#define cublasGemmEx            hipblasGemmEx
#define CUBLAS_STATUS_SUCCESS   HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N             HIPBLAS_OP_N
#define CUBLAS_OP_T             HIPBLAS_OP_T
#define CUBLAS_COMPUTE_32F      HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT     HIPBLAS_GEMM_DEFAULT

// Data types for GEMM
#define CUDA_R_16BF             HIP_R_16BF

// Warp shuffle mask for HIP (64-bit required)
// HIP requires 64-bit masks for __shfl_* functions
#define WARP_FULL_MASK 0xffffffffffffffffULL

#else

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// On CUDA, use the standard 32-bit mask
#define WARP_FULL_MASK 0xffffffff

#endif
