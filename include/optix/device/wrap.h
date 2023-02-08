#pragma once

#include <cuda_runtime.h>

namespace optix {

struct Wrap {
    static __forceinline__ __device__ float2
    squareToUniformDisk(const float2& sample) {
        float r = sqrtf(sample.x);
        float theta = 2.0f * M_PIf * sample.y;
        return make_float2(r * cosf(theta), r * sinf(theta));
    }

    static __forceinline__ __device__ float3
    squareToCosineHemisphere(const float2& sample) {
        float2 p = squareToUniformDisk(sample);
        float z = sqrtf(1 - p.x * p.x - p.y * p.y);
        return make_float3(p.x, p.y, z);
    }
};

}  // namespace optix