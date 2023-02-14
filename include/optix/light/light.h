#pragma once

#include "optix/light/area.h"
#include "optix/light/point.h"
#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/shape/shape.h"
#include <cuda_runtime.h>

namespace optix {

struct Light {
    Light() {}

    enum class Type : int { POINT = 0, AREA = 1 };

    Type type;

    union {
        Point point;
        Area area;
    };

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int seed,
                    LightSampleRecord& dRec) const {
        if (type == Type::POINT) {
            return point.sampleDirection(its, seed, dRec);
        }
        else if (type == Type::AREA) {
            return area.sampleDirection(its, seed, dRec);
        }
        return make_float3(0.f);
    }

    /**
     * @brief Query the probability density of @ref sampleDirection()
     */
    float pdfDirection(const LightSampleRecord& dRec) const {
        if (type == Type::POINT) {
            return point.pdfDirection(dRec);
        }
        else if (type == Type::AREA) {
            return area.pdfDirection(dRec);
        }
        return 0;
    }

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
        if (type == Type::POINT) {
            return point.eval(its, wi);
        }
        else if (type == Type::AREA) {
            return area.eval(its, wi);
        }
        return make_float3(0.f);
    }
};

struct LightBuffer {
    Light* lights;
    int light_num;

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleLightDirection(const Intersection& its, unsigned int& seed,
                         LightSampleRecord& dRec) const {
        if (light_num == 0) {
            return make_float3(0.f);
        }
        else {
            float light_pdf = 1.f / light_num;

            // Randomly pick a light
            int index = min((int)(rnd(seed) * light_num), light_num - 1);
            const Light& light = lights[index];

            float3 spec = light.sampleDirection(its, seed, dRec);

            spec = spec * (float)light_num;
            dRec.pdf = dRec.pdf * light_pdf;
            return spec;
        }
    }

    SUTIL_INLINE SUTIL_HOSTDEVICE float
    pdfLightDirection(int idx, const LightSampleRecord& dRec) const {
        return lights[idx].pdfDirection(dRec) / light_num;
    }
};

}  // namespace optix