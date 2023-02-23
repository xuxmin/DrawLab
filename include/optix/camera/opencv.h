#pragma once

#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/math/matrix.h"
#include <cuda_runtime.h>

namespace optix {

struct OpencvCamera {

    // the inverse matrix of extrinsic
    float invExt[16];

    // intrinsic
    float3 eye;
    float cx;
    float cy;
    float fx;
    float fy;


#ifdef __CUDACC__
    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleRay(const float2 screen, const uint3 launch_idx, unsigned int& seed,
              const int sample_idx, const int spp, float3& ray_origin,
              float3& ray_direction) const {

        float2 pixel = make_float2(((float)launch_idx.x - cx) / fx,
                                   ((float)launch_idx.y - cy) / fy);

        int sqrt_num_samples = (int)sqrtf((float)spp);
        float2 jitter_scale = make_float2(1.f / fx, 1.f / fy) / (float)sqrt_num_samples;

        unsigned int x = sample_idx % sqrt_num_samples;
        unsigned int y = sample_idx / sqrt_num_samples;
        float2 jitter = make_float2(x + rnd(seed), y + rnd(seed));

        const float2 subpixel_jitter = jitter * jitter_scale;
        float2 d = pixel + subpixel_jitter;

		float4 head = make_float4(d, 1.0, 1.0);
		head = Matrix4x4(invExt) * head;		// convert pixel coord to world coordinate

        ray_origin = eye;
        ray_direction = normalize(make_float3(head) - eye);
    }
#endif

};

}  // namespace optix
