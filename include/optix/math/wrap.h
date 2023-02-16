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

    static __forceinline__ __device__ float
    squareToCosineHemispherePdf(const float3& v) {
        return v.z <= 0 ? 0 : v.z * M_INV_PI;
    }

    static __forceinline__ __device__ float3
    squareToUniformTriangle(const float2& sample) {
        float t = sqrtf(1 - sample.x);
        return make_float3(1 - t, t * sample.y, t - t * sample.y);
    }

    static __forceinline__ __device__ float3
    squareToBeckmann(const float2& sample, float alpha) {
        float theta = atanf(sqrtf(-alpha * alpha * logf(1 - sample.x)));
        float phi = 2 * M_PIf * sample.y;
        return make_float3(sinf(theta) * cosf(phi),
                           sinf(theta) * sinf(phi),
                           cosf(theta));
    }

    static __forceinline__ __device__ float
    squareToBeckmannPdf(const float3& m, float alpha) {
        if (m.z <= 0)
            return 0;

        float cosTheta = m.z;
        float tanTheta_2 = (1 - cosTheta * cosTheta) / (cosTheta * cosTheta);
        float p = M_INV_TWOPI * (2 * expf(-tanTheta_2 / (alpha * alpha))) /
                (alpha * alpha * cosTheta * cosTheta * cosTheta);
        return p;
    }
};

}  // namespace optix