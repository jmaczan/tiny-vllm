#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

void softmax(__nv_bfloat16 *input, int num_tokens);
void softmaxDecode(__nv_bfloat16 *input, int seq_len);

constexpr int kHeads = 32;
constexpr int kDecodeStride = 2048;

bool checkCuda(cudaError_t status, const char *step)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << step << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool runCase(const char *name, const std::vector<float> &scores, bool decode)
{
    const int length = static_cast<int>(scores.size());
    const int rows = decode ? kHeads : length * kHeads;
    const int stride = decode ? kDecodeStride : length;
    std::vector<__nv_bfloat16> host(rows * stride, __nv_bfloat16(0.0f));
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < length; ++col)
            host[row * stride + col] = __nv_bfloat16(scores[col]);

    __nv_bfloat16 *device = nullptr;
    if (!checkCuda(cudaMalloc(&device, host.size() * sizeof(host[0])), "cudaMalloc"))
        return false;
    bool ok = checkCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(host[0]),
                                   cudaMemcpyHostToDevice), "input copy");
    if (ok)
    {
        if (decode)
            softmaxDecode(device, length);
        else
            softmax(device, length);
        ok = checkCuda(cudaDeviceSynchronize(), "softmax kernel") &&
             checkCuda(cudaMemcpy(host.data(), device, host.size() * sizeof(host[0]),
                                  cudaMemcpyDeviceToHost), "output copy");
    }
    checkCuda(cudaFree(device), "cudaFree");
    if (!ok)
        return false;

    float maxScore = -std::numeric_limits<float>::infinity();
    for (float score : scores)
        maxScore = std::fmax(maxScore, score);
    float denominator = 0.0f;
    if (std::isfinite(maxScore))
        for (float score : scores)
            if (std::isfinite(score))
                denominator += std::exp(score - maxScore);

    for (int row = 0; row < rows; ++row)
    {
        float sum = 0.0f;
        for (int col = 0; col < length; ++col)
        {
            const float actual = static_cast<float>(host[row * stride + col]);
            const float expected = std::isfinite(scores[col])
                                       ? std::exp(scores[col] - maxScore) / denominator
                                       : 0.0f;
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.01f)
            {
                std::cerr << name << " row " << row << " column " << col
                          << ": got " << actual << ", expected " << expected << '\n';
                return false;
            }
            sum += actual;
        }
        const float expectedSum = denominator > 0.0f ? 1.0f : 0.0f;
        if (std::fabs(sum - expectedSum) > 0.01f)
        {
            std::cerr << name << " row " << row << ": sum " << sum << '\n';
            return false;
        }
    }
    std::cout << "PASS " << name << '\n';
    return true;
}

int main()
{
    const float masked = -std::numeric_limits<float>::infinity();
    bool ok = true;
    ok &= runCase("prefill causal row", {1.0f, masked, masked, masked}, false);
    ok &= runCase("prefill masked subtree", {masked, masked, 2.0f, 1.0f}, false);
    ok &= runCase("prefill entirely masked", {masked, masked, masked, masked}, false);
    ok &= runCase("decode masked subtree", {masked, masked, 2.0f, 1.0f}, true);
    ok &= runCase("decode entirely masked", {masked, masked, masked, masked}, true);
    return ok ? 0 : 1;
}
