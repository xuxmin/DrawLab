#pragma once

#include "optix/shape/shape.h"
#include <cuda_runtime.h>

namespace optix {

struct EnvAccel {
    // alias map
    float q;
    unsigned int alias;
    // associated pdf
    float pdf;
};

struct Envmap {
    cudaTextureObject_t env_tex;
    EnvAccel* env_accel;
    int2 env_size;
    bool visual;

#ifdef __CUDACC__
    // direction to environment map coordinates
    static SUTIL_INLINE SUTIL_HOSTDEVICE float2
    environment_coords(const float3& dir) {
        const float u = atan2f(dir.z, dir.x) * (float)(0.5 / M_PIf) + 0.5f;
        const float v = acosf(fmax(fminf(dir.y, 1.0f), -1.0f)) * M_INV_PI;
        return make_float2(u, v);
    }

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int& seed,
                    LightSampleRecord& dRec) const {
        const float3 xi = make_float3(rnd(seed), rnd(seed), rnd(seed));

        // importance sample an envmap pixel using an alias map
        const unsigned int size = env_size.x * env_size.y;
        const unsigned int idx = min((unsigned int)(xi.x * (float)size), size - 1);
        unsigned int env_idx;
        float xi_y = xi.y;
        if (xi_y < env_accel[idx].q) {
            env_idx = idx;
            xi_y /= env_accel[idx].q;
        }
        else {
            env_idx = env_accel[idx].alias;
            xi_y = (xi_y - env_accel[idx].q) / (1.0f - env_accel[idx].q);
        }

        const unsigned int py = env_idx / env_size.x;
        const unsigned int px = env_idx % env_size.x;
        dRec.pdf = env_accel[env_idx].pdf;

        // uniformly sample spherical area of pixel
        const float u = (float)(px + xi_y) / (float)env_size.x;
        const float phi = u * 2.0 * M_PIf - M_PIf;
        const float step_theta = M_PIf / (float)env_size.y;
        const float theta0 = (float)(py)*step_theta;
        const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
        const float theta = acosf(cos_theta);
        const float sin_theta = sinf(theta);
        float3 dir = make_float3(cosf(phi) * sin_theta, cos_theta,
                                 sinf(phi) * sin_theta);

        dRec.o = its.p;
        dRec.d = dir;
        dRec.dist = 1e16;
        dRec.delta = false;
        dRec.n = -dir;
        dRec.mesh = nullptr;

        // lookup filtered value
        const float v = theta * M_INV_PI;
        const float3 val = make_float3(tex2D<float4>(env_tex, u, v));

        float cosTheta = fmaxf(dot(its.sn, dRec.d), 0.f);
        return cosTheta > 0.f ? cosTheta * val / dRec.pdf : make_float3(0.f);
    }

    float pdfDirection(const LightSampleRecord& dRec) const {
        const float2 uv = environment_coords(dRec.d);
        int x = min((int)(uv.x * (float)env_size.x), env_size.x - 1);
        int y = min((int)(uv.y * (float)env_size.y), env_size.y - 1);
        float pdf = env_accel[y * env_size.x + x].pdf;
        return pdf;
    }

    SUTIL_INLINE SUTIL_HOSTDEVICE float3 eval(const Intersection& its,
                                              float3 wi) const {
        const float2 uv = environment_coords(wi);
        const float3 val = make_float3(tex2D<float4>(env_tex, uv.x, uv.y));
        return val;
    }
#endif
};

}  // namespace optix