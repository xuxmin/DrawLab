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

    static __forceinline__ __device__ float
    squareToGGXPdf(const float3& m, float2 axay) {
        if (m.z <= 0)
            return 0;
        
        // m is half vector
        float3 vhalf = m;
        vhalf.x /= axay.x;
        vhalf.y /= axay.y;

        float len2 = dot(vhalf, vhalf);
        float D = 1.f / (M_PIf * axay.x * axay.y * len2 * len2);
        return D;
    }


    static __forceinline__ __device__ float3
    squareToGGX(const float2& sample, float2 axay) {
		float z1 = sample.x;
		float z2 = sample.y;
		float x = axay.x * sqrtf(z1) / sqrtf(1 - z1) * cos(2 * M_PIf * z2);
		float y = axay.y * sqrtf(z1) / sqrtf(1 - z1) * sin(2 * M_PIf * z2);
		return normalize(make_float3(-x, -y, 1));
    }
};

}  // namespace optix