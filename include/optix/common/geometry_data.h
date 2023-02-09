#pragma once

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
    float3 o;   // position
    float3 d;   // direction
    float3 n;   // normal in dst

    float pdf;
    float dist;
    float delta;
};

}  // namespace optix
