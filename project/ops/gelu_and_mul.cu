#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ bool is_16byte_aligned(const void *ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// GELU 计算逻辑
template <typename T>
__device__ __forceinline__ T gelu_kernel_impl(const T &x)
{
    float f = (float)x;
    // M_SQRT1_2 是 1/sqrt(2)
    constexpr float ALPHA = 0.70710678118654752440f;
    return (T)(f * 0.5f * (1.0f + erff(f * ALPHA)));
}

template <typename scalar_t>
__global__ void gelu_and_mul_kernel(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ input,
    const int d)
{

    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    const int64_t token_idx = blockIdx.x;

    const scalar_t *x_ptr = input + token_idx * 2 * d;
    const scalar_t *y_ptr = x_ptr + d;
    scalar_t *out_ptr = out + token_idx * d;

    const bool aligned = is_16byte_aligned(x_ptr) &&
                         is_16byte_aligned(y_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && d % VEC_SIZE == 0)
    {
        const int4 *x_vec = reinterpret_cast<const int4 *>(x_ptr);
        const int4 *y_vec = reinterpret_cast<const int4 *>(y_ptr);
        int4 *out_vec = reinterpret_cast<int4 *>(out_ptr);
        const int num_vecs = d / VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x)
        {
            int4 vx = x_vec[i];
            int4 vy = y_vec[i];
            int4 vr;

            scalar_t *sx = reinterpret_cast<scalar_t *>(&vx);
            scalar_t *sy = reinterpret_cast<scalar_t *>(&vy);
            scalar_t *sr = reinterpret_cast<scalar_t *>(&vr);

#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++)
            {
                sr[j] = sx[j] * gelu_kernel_impl(sy[j]);
            }
            out_vec[i] = vr;
        }
    }
    else
    {
        // 标量回退路径
        for (int i = threadIdx.x; i < d; i += blockDim.x)
        {
            scalar_t x = x_ptr[i];
            scalar_t y = y_ptr[i];
            out_ptr[i] = x * gelu_kernel_impl(y);
        }
    }
}

// C++ 封装函数
void launch_gelu_and_mul(torch::Tensor &out, torch::Tensor &input)
{
    int64_t num_tokens = input.size(0);
    int d = out.size(-1);

    dim3 grid(num_tokens);
    dim3 block(std::min(d, 512));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "gelu_and_mul_kernel", ([&]
                                                                                     { gelu_and_mul_kernel<scalar_t><<<grid, block>>>(
                                                                                           out.data_ptr<scalar_t>(),
                                                                                           input.data_ptr<scalar_t>(),
                                                                                           d); }));
}