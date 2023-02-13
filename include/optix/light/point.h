#pragma once

#include "optix/shape/shape.h"
#include <cuda_runtime.h>

namespace optix {

struct Point {
    float3 intensity;
    float3 position;

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int seed,
                    DirectionSampleRecord& dRec) const {
        const float3 intensity = intensity;
        const float3 light_pos = position;

        dRec.o = its.p;
        dRec.d = normalize(light_pos - its.p);
        dRec.n = make_float3(0.f);
        dRec.delta = true;
        dRec.dist = length(light_pos - its.p);
        dRec.pdf = 1;
        float inv_dist = (float)1.0 / dRec.dist;
        return intensity * inv_dist * inv_dist;
    }

    float pdfDirection(const DirectionSampleRecord& dRec) const { return 0; }

    /**
     * @brief Given a ray-surface intersection, return the emitted
     * radiance or importance traveling along the reverse direction
     *
     * @param its The intersection of ray and light.
     * @param wi The ray direction from light to surface point
     * @return  The emitted radiance or importance
     */
    SUTIL_INLINE SUTIL_HOSTDEVICE float3 eval(const Intersection& its,
                                              float3 wi) const {
        return make_float3(0.f);
    }
};

}  // namespace optix