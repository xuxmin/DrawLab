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

}  // namespace optix
