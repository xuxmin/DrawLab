#pragma once

#include "optix/math/vec_math.h"
#include "optix/math/random.h"
#include "optix/math/wrap.h"
#include "optix/shape/mesh.h"

namespace optix {

struct Shape {
    enum Type {
        TRIANGLE_MESH = 0,
    };

    Type type;

    union {
        TriangleMesh triangle_mesh;
    };
};

struct Intersection {
    const TriangleMesh* mesh;
    float3 sn;  // shading normal
    float3 gn;  // geometry normal
    float2 uv;
    float3 p;
    int light_idx;
};

struct DirectionSampleRecord {
    const TriangleMesh* mesh;

    float3 o;  // position
    float3 d;  // direction
    float3 n;  // normal in dst

    float pdf;
    float dist;
    bool delta;

    DirectionSampleRecord() {}

    DirectionSampleRecord(const float3& ori, const float3& dst, const float3 n,
                          const TriangleMesh* mesh)
        : o(ori), n(n), mesh(mesh) {
        float3 vec = dst - ori;
        d = normalize(vec);
        dist = length(vec);
        delta = false;
    }
};

}  // namespace optix
