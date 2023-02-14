#pragma once

#include "optix/shape/shape.h"
#include <cuda_runtime.h>

namespace optix {

struct Area {
    TriangleMesh triangle_mesh;
    float3 intensity;

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int seed,
                    LightSampleRecord& dRec) const {
        float3 position, normal;
        triangle_mesh.samplePosition(seed, position, normal);

        float3 vec = position - its.p;
        dRec.o = its.p;
        dRec.d = normalize(vec);
        dRec.dist = length(vec);
        dRec.delta = false;
        dRec.n = normal;
        dRec.mesh = nullptr;

        float pA = triangle_mesh.pdfPosition();
        float dp = fmaxf(dot(normal, -dRec.d), 0.f);
        float pw = dp != 0 ? pA * dRec.dist * dRec.dist / dp : 0;
        dRec.pdf = pw;

        return pw != 0 ? intensity * fmaxf(dot(its.sn, dRec.d), 0.f) / pw :
                         make_float3(0.f);
    }

    float pdfDirection(const LightSampleRecord& dRec) const {
        float pA = triangle_mesh.pdfPosition();
        float dp = fmaxf(dot(dRec.n, -dRec.d), 0.f);
        float pw = dp != 0 ? pA * dRec.dist * dRec.dist / dp : 0;
        return pw;
    }

    SUTIL_INLINE SUTIL_HOSTDEVICE float3 eval(const Intersection& its,
                                              float3 wi) const {
        float cosTheta = dot(its.sn, wi);
        return cosTheta > 0.f ? intensity : make_float3(0.f);
    }
};

}  // namespace optix