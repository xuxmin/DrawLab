#pragma once

#include "optix/camera/virtual.h"
#include <cuda_runtime.h>

namespace optix {

struct Camera {
    enum Type {
        VIRTUAL = 0,
    };

    Type type;

    union {
        Virtual virtual_cam;
    };

    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleRay(const float2 screen, const uint3 launch_idx, unsigned int& seed,
              const int sample_idx, const int spp, float3& ray_origin,
              float3& ray_direction) const {
        if (type == VIRTUAL) {
            virtual_cam.sampleRay(screen, launch_idx, seed, sample_idx, spp,
                                  ray_origin, ray_direction);
        }
    }
};

}  // namespace optix
