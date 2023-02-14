#pragma once

#include "optix/math/vec_math.h"
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

struct LightSampleRecord {
    const TriangleMesh* mesh;

    float3 o;  // position
    float3 d;  // direction
    float3 n;  // normal in dst

    float pdf;
    float dist;
    bool delta;

    LightSampleRecord() {}

    LightSampleRecord(const float3& ori, const float3& dst, const float3 n,
                      const TriangleMesh* mesh)
        : o(ori), n(n), mesh(mesh) {
        float3 vec = dst - ori;
        d = normalize(vec);
        dist = length(vec);
        delta = false;
    }
};

// BSDF sample record in the prev path
struct BSDFSampleRecord {
    float3 fr;  // eval() / pdf() * cos(theta)
    float eta;
    float3 p;   // surface point position
    float3 wo;  // sampled direction in world coordinate.
    float pdf;
    bool is_diffuse;

    BSDFSampleRecord()
        : fr(make_float3(1.f)), eta(1.f), pdf(0.f), is_diffuse(false) {}
};

enum EMeasure { EUnknownMeasure = 0, ESolidAngle, EDiscrete };

struct BSDFQueryRecord {
    /// Reference to the underlying surface interaction
    const Intersection& its;
    /// Incident direction (in the local frame)
    float3 wi;
    /// Outgoing direction (in the local frame)
    float3 wo;
    /// Relative refractive index in the sampled direction
    float eta;
    /// Measure associated with the sample
    EMeasure measure;

    BSDFQueryRecord(const Intersection& its, const float3& wi)
        : its(its), wi(wi), eta(1.f), measure(EUnknownMeasure) {}

    BSDFQueryRecord(const Intersection& its, const float3& wi, const float3& wo,
                    EMeasure measure)
        : its(its), wi(wi), wo(wo), eta(1.f), measure(measure) {}
};

}  // namespace optix
