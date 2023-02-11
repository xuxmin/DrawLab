#pragma once

#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/wrap.h"

namespace optix {

struct GeometryData {
    enum Type {
        TRIANGLE_MESH = 0,
        SPHERE = 1,
    };

    struct TriangleMesh {
        float3* positions;
        int3* indices;
        float3* normals;
        float2* texcoords;

        float* cdf;
        float face_num;
        float pdf;

        /// Sample a face base on the surface area
        SUTIL_INLINE SUTIL_HOSTDEVICE int sampleFace(float sampleValue) const {
            int l = 0;
            int r = face_num;
            while (l < r) {
                int mid = (l + r) / 2;
                if (cdf[mid] > sampleValue) {
                    r = mid;
                }
                else if (cdf[mid] < sampleValue) {
                    l = mid + 1;
                }
                else {
                    r = mid;
                    break;
                }
            }
            int index = max(l - 1, 0);
            return index;
        }

        SUTIL_INLINE SUTIL_HOSTDEVICE void
        samplePosition(unsigned int seed, float3& position,
                       float3& normal) const {
            // int f = sampleFace(rnd(seed));
            int f = min(int(rnd(seed) * face_num), (int)face_num-1);
            const int3 index = indices[f];
            const float3& p0 = positions[index.x];
            const float3& p1 = positions[index.y];
            const float3& p2 = positions[index.z];

            float3 bary = Wrap::squareToUniformTriangle(
                make_float2(rnd(seed), rnd(seed)));
            position = bary.x * p0 + bary.y * p1 + bary.z * p2;

            if (normals) {
                const float3& n0 = normals[index.x];
                const float3& n1 = normals[index.y];
                const float3& n2 = normals[index.z];
                normal = bary.x * n0 + bary.y * n1 + bary.z * n2;
            }
            else {
                normal = cross(p1 - p0, p2 - p0);
            }
            normal = normalize(normal);
        }
        
        SUTIL_INLINE SUTIL_HOSTDEVICE float pdfPosition() const {
            return pdf;
        }
    };

    Type type;

    union {
        TriangleMesh triangle_mesh;
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
