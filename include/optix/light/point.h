#pragma once

#include "optix/shape/shape.h"
#include <cuda_runtime.h>

namespace optix {

struct LightSampleRecord;

struct Point {
    float3 intensity;
    float3 position;

#ifdef __CUDACC__
    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int& seed,
                    LightSampleRecord& dRec) const {
        dRec.o = its.p;
        dRec.d = normalize(position - its.p);
        dRec.n = make_float3(0.f);
        dRec.delta = true;
        dRec.dist = length(position - its.p);
        dRec.pdf = 1;
        float inv_dist = (float)1.0 / dRec.dist;
        float cosTheta = fmaxf(dot(its.sn, dRec.d), 0.f);
        return cosTheta > 0.f ? cosTheta * intensity * inv_dist * inv_dist : make_float3(0.f);
    }

    float pdfDirection(const LightSampleRecord& dRec) const { return 0; }

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
#endif
};

}  // namespace optix