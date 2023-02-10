//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once
#include "optix/common/geometry_data.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include <cuda_runtime.h>

namespace optix {

struct Light {
    Light() {}

    enum class Type : int { POINT = 0, AREA = 1 };

    struct Point {
        float3 intensity CONST_STATIC_INIT({1.0f, 1.0f, 1.0f});
        float3 position CONST_STATIC_INIT({});
    };

    struct Area {
        GeometryData::TriangleMesh triangle_mesh;
        float3 intensity;
    };

    Type type;

    union {
        Point point;
        Area area;
    };

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleDirection(const Intersection& its, unsigned int seed,
                    DirectionSampleRecord& dRec) const {
        if (type == Type::POINT) {
            const float3 intensity = point.intensity;
            const float3 light_pos = point.position;

            dRec.o = its.p;
            dRec.d = normalize(light_pos - its.p);
            dRec.n = make_float3(0.f);
            dRec.delta = true;
            dRec.dist = length(light_pos - its.fp);
            dRec.pdf = 1;
            float inv_dist = (float)1.0 / dRec.dist;
            return intensity * inv_dist * inv_dist;
        }
        else {
            float3 position, normal;
            area.triangle_mesh.samplePosition(seed, position, normal);

            float3 vec = position - its.p;
            dRec.o = its.p;
            dRec.d = normalize(vec);
            dRec.dist = length(vec);
            dRec.delta = false;
            dRec.n = normal;
            dRec.mesh = nullptr;

            float pA = area.triangle_mesh.pdfPosition();
            float dp = fmaxf(dot(normal, -dRec.d), 0.f);
            float pw = dp != 0 ? pA * dRec.dist * dRec.dist / dp : 0;
            dRec.pdf = pw;

            return pw != 0 ?
                       area.intensity * fmaxf(dot(its.sn, dRec.d), 0.f) / pw :
                       make_float3(0.f);
        }
    }

    /**
     * @brief Query the probability density of @ref sampleDirection()
     */
    float pdfDirection(const DirectionSampleRecord& dRec) const {
        if (type == Type::POINT) {
            return 0;
        }
        else if (type == Type::AREA) {
            float pA = area.triangle_mesh.pdfPosition();
            float dp = fmaxf(dot(dRec.n, -dRec.d), 0.f);
            float pw = dp != 0 ? pA * dRec.dist * dRec.dist / dp : 0;
            return pw;
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
            return make_float3(0.f);
        }
        else if (type == Type::AREA) {
            float cosTheta = dot(its.sn, wi);
            return cosTheta > 0.f ? area.intensity : make_float3(0.f);
        }
        return make_float3(0.f);
    }
};

struct LightData {
    Light* lights;
    int light_num;

    SUTIL_INLINE SUTIL_HOSTDEVICE float3
    sampleLightDirection(const Intersection& its, unsigned int seed,
                         DirectionSampleRecord& dRec) const {
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
    pdfLightDirection(int idx, const DirectionSampleRecord& dRec) const {
        return lights[idx].pdfDirection(dRec) / light_num;
    }
};

}  // namespace optix