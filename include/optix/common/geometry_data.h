#pragma once

#include "optix/common/vec_math.h"

namespace optix {

struct GeometryData {
    enum Type {
        TRIANGLE_MESH = 0,
        SPHERE = 1,
    };

    struct TriangleMesh {
        int3* indices;
        float3* positions;
        float3* normals;
        float2* texcoords;
    };

    struct Sphere {
        float3 center;
        float radius;
    };

    Type type;

    union {
        TriangleMesh triangle_mesh;
        Sphere sphere;
    };
};

struct Intersection {
    const GeometryData::TriangleMesh* mesh;
    float3 sn;  // shading normal
    float3 gn;  // geometry normal
    float2 uv;
    float3 p;
    float3 bp;
    float3 fp;
    int light_idx;
};

struct DirectionSampleRecord {
    const GeometryData::TriangleMesh* mesh;

    float3 o;  // position
    float3 d;  // direction
    float3 n;  // normal in dst

    float pdf;
    float dist;
    bool delta;

    DirectionSampleRecord() {}

    DirectionSampleRecord(const float3& ori, const float3& dst, const float3 n,
                          const GeometryData::TriangleMesh* mesh)
        : o(ori), n(n), mesh(mesh) {
        float3 vec = dst - ori;
        d = normalize(vec);
        dist = length(vec);
        delta = false;
    }
};

}  // namespace optix
