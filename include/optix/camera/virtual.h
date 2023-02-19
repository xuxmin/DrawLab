#pragma once

#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include <cuda_runtime.h>

namespace optix {

struct Virtual {
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    float3 looat;
    float3 up;
    float fov;

    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleRay(const float2 screen, const uint3 launch_idx, unsigned int& seed,
              const int sample_idx, const int spp, float3& ray_origin,
              float3& ray_direction) const {
        float2 inv_screen = 1.0f / screen * 2.f;
        float2 pixel = make_float2((float)launch_idx.x, (float)launch_idx.y);
        pixel = pixel * inv_screen - 1.f;

        int sqrt_num_samples = (int)sqrtf((float)spp);
        float2 jitter_scale = inv_screen / (float)sqrt_num_samples;

        unsigned int x = sample_idx % sqrt_num_samples;
        unsigned int y = sample_idx / sqrt_num_samples;
        float2 jitter = make_float2(x + rnd(seed), y + rnd(seed));

        const float2 subpixel_jitter = jitter * jitter_scale;
        float2 d = pixel + subpixel_jitter;
        ray_origin = eye;
        ray_direction = normalize(d.x * U + d.y * (-V) + W);
    }
};

}  // namespace optix
