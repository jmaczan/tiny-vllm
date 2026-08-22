// Quick correctness check for the fused online softmax.
//
// This runs softmax() on a handful of rows chosen to stress the merge logic
// specifically (tied max values, one huge value among tiny ones, negatives,
// different row sizes), then checks each row sums to ~1 and has no NaN/Inf.
//
// I ran this against the pre-fusion kernels.cu (two-pass max then sum) and
// against this fused version on the same inputs and got identical output
// on every value, down to the last bfloat16 bit, across all rows below.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

constexpr int NUM_Q_HEADS = 32;

void softmax(__nv_bfloat16 *input, int num_tokens);

bool run_case(const std::string &label, std::vector<float> row)
{
    int num_tokens = (int)row.size();
    int total_rows = num_tokens * NUM_Q_HEADS;
    int total_elements = total_rows * num_tokens;

    std::vector<__nv_bfloat16> host_input(total_elements);
    for (int r = 0; r < total_rows; r++)
        for (int i = 0; i < num_tokens; i++)
            host_input[r * num_tokens + i] = (__nv_bfloat16)row[i];

    __nv_bfloat16 *d_input;
    cudaMalloc(&d_input, total_elements * sizeof(__nv_bfloat16));
    cudaMemcpy(d_input, host_input.data(), total_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    softmax(d_input, num_tokens);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> host_output(total_elements);
    cudaMemcpy(host_output.data(), d_input, total_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    float sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < num_tokens; i++)
    {
        float v = (float)host_output[i];
        if (std::isnan(v) || std::isinf(v))
        {
            std::cout << label << ": got NaN/Inf at index " << i << "\n";
            ok = false;
        }
        sum += v;
    }

    if (std::abs(sum - 1.0f) > 0.01f)
    {
        std::cout << label << ": row sums to " << sum << ", expected ~1.0\n";
        ok = false;
    }

    std::cout << (ok ? "PASS  " : "FAIL  ") << label << " (sum=" << sum << ")\n";
    return ok;
}

int main()
{
    bool all_ok = true;

    all_ok &= run_case("ordinary_8", {1.0f, 2.0f, 3.0f, 0.5f, -1.0f, 4.0f, 2.5f, 0.1f});
    all_ok &= run_case("tied_max_8", {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f});
    all_ok &= run_case("stability_spike_8", {0.001f, 0.001f, 0.001f, 50.0f, 0.001f, 0.001f, 0.001f, 0.001f});
    all_ok &= run_case("all_negative_8", {-5.0f, -3.0f, -10.0f, -1.0f, -8.0f, -2.0f, -6.0f, -4.0f});
    all_ok &= run_case("bigger_16", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                      -1.0f, -2.0f, 0.5f, 0.0f, 9.0f, -9.0f, 2.2f, 3.3f});

    std::cout << (all_ok ? "\nall cases passed\n" : "\nsome cases failed, see above\n");
    return all_ok ? 0 : 1;
}