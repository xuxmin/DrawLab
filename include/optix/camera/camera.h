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
    sampleRay(const int w, const int h, const uint3 idx, unsigned int seed,
              float3& ray_origin, float3& ray_direction) const {
                if (type == VIRTUAL) {
            virtual_cam.sampleRay(w, h, idx, seed, ray_origin, ray_direction);
        }
    }
};

}  // namespace optix
