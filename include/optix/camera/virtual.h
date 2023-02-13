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
    sampleRay(const int w, const int h, const uint3 idx, unsigned int seed,
              float3& ray_origin, float3& ray_direction) const {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d =
            2.0f * make_float2((static_cast<float>(idx.x) + subpixel_jitter.x) /
                                   static_cast<float>(w),
                               (static_cast<float>(idx.y) + subpixel_jitter.y) /
                                   static_cast<float>(h)) -
            1.0f;
        ray_direction = normalize(d.x * U + d.y * (-V) + W);
        ray_origin = eye;
    }
};

}  // namespace optix
