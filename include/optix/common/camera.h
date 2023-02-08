#pragma once
#include <cuda_runtime.h>
#include "optix/common/vec_math.h"


namespace optix{

struct Camera {
    enum Type {
        VIRTUAL = 0,
    };

    struct Virtual {
        float3 eye;
        float3 U;
        float3 V;
        float3 W;

        float3 looat;
        float3 up;
        float fov;
    };

    Type type;

    union {
        Virtual virtual_cam;
    };

    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleRay(const int width, const int height, const uint3 launch_idx,
              float3& ray_origin, float3& ray_direction, unsigned int& seed) {
        const int w = width;
        const int h = height;

        seed = launch_idx.y * w + launch_idx.x;

        if (type == Camera::VIRTUAL) {
            const float3 eye = virtual_cam.eye;
            const float3 U = virtual_cam.U;
            const float3 V = -virtual_cam.V;
            const float3 W = virtual_cam.W;

            float2 d =
                make_float2((float)launch_idx.x / w, (float)launch_idx.y / h);
            d = 2.f * d - 1.f;
            ray_direction = normalize(d.x * U + d.y * V + W);
            ray_origin = eye;
        }
    };
};

} // namespace optix
