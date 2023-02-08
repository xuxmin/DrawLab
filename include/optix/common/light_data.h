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
#include <cuda_runtime.h>
#include "optix/common/vec_math.h"
#include "optix/device/random.h"

namespace optix {

struct Light {
    Light() {}

    enum class Type : int { POINT = 0, AREA = 1 };

    struct Point {
        float3 intensity CONST_STATIC_INIT({1.0f, 1.0f, 1.0f});
        float3 position CONST_STATIC_INIT({});
    };

    struct Area {
        float3 corner;
        float3 v1, v2;
        float3 normal;
        float3 emission;
    };

    Type type;

    union {
        Point point;
        Area area;
    };

    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleDirection(const float3& surface_pos, unsigned int seed,
                         float3& direction, float& pdf,
                         float3& spectrum) const {
        if (type == Type::POINT) {
            const float3 intensity = point.intensity;
            const float3 light_pos = point.position;

            float inv_dist = 1.0 / length(surface_pos - light_pos);

            // do not normalize it!
            direction = light_pos - surface_pos;
            pdf = 1;
            spectrum = intensity * inv_dist * inv_dist;
        }
    }
};

struct LightData {
    Light* lights;
    int light_num;

    SUTIL_INLINE SUTIL_HOSTDEVICE void
    sampleLightDirection(const float3& surface_pos, unsigned int seed,
                         float3& direction, float& pdf,
                         float3& spectrum) const {
        if (light_num == 0) {
            spectrum = make_float3(0.f);
        }
        else {
            float light_pdf = 1.f / light_num;

            // Randomly pick a light
            int index = min((int)(rnd(seed) * light_num), light_num - 1);
            const Light& light = lights[index];

            light.sampleDirection(surface_pos, seed, direction, pdf, spectrum);

            spectrum = spectrum * (float)light_num;
            pdf = pdf * light_pdf;
        }
    }
};

}  // namespace optix