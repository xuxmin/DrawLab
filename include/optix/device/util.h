#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "optix/common/optix_params.h"
#include "optix/device/intersection_refinement.h"


namespace optix {

static __forceinline__ __device__ void* unpackPointer(unsigned int i0,
                                                      unsigned int i1) {
    const unsigned long long uptr =
        static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0,
                                                   unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T> static __forceinline__ __device__ T* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

static __forceinline__ __device__ bool
traceOcclusion(OptixTraversableHandle handle, float3 ray_origin,
               float3 ray_direction, float tmin, float tmax) {
    unsigned int occluded = 0u;
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION,  // SBT offset
               RAY_TYPE_COUNT,      // SBT stride
               RAY_TYPE_OCCLUSION,  // missSBTIndex
               occluded);
    return occluded;
}

static __forceinline__ __device__ float mis_weight(float pdf_a, float pdf_b) {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    return pdf_a / (pdf_a + pdf_b);
}

static __forceinline__ __device__ Intersection getHitData() {
    Intersection its;
    // ------------------------------------------------------------------
    // Gather basic hit information
    // ------------------------------------------------------------------
    
    const HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const GeometryData::TriangleMesh& mesh =
        reinterpret_cast<const GeometryData::TriangleMesh&>(
            rt_data->geometry_data.triangle_mesh);
    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh.indices[prim_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const float3 v0 = mesh.positions[index.x];
    const float3 v1 = mesh.positions[index.y];
    const float3 v2 = mesh.positions[index.z];
    float3 gN = normalize(cross(v1 - v0, v2 - v0));     // geometry normal
    float3 sN = gN;                                     // shading normal

    if (mesh.normals) {
        const float3 n0 = mesh.normals[index.x];
        const float3 n1 = mesh.normals[index.y];
        const float3 n2 = mesh.normals[index.z];
        sN = (1.f - u - v) * n0 + u * n1 + v * n2;
    }

    // face-forward and normalize normals
    gN = faceforward(gN, -ray_dir, gN);
    if (dot(gN, sN) < 0.f)
        sN -= 2.f * dot(gN, sN) * gN;
    sN = normalize(sN);

    // ------------------------------------------------------------------
    // Compute texcoords
    // ------------------------------------------------------------------
    float2 texcoord = make_float2(0.f);
    if (mesh.texcoords) {
        const float2 t0 = mesh.texcoords[index.x];
        const float2 t1 = mesh.texcoords[index.y];
        const float2 t2 = mesh.texcoords[index.z];
        texcoord = (1.f - u - v) * t0 + u * t1 + v * t2;
    }

    const float3 hitpoint = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    float3 bp, fp;

    refine_and_offset_hitpoint(hitpoint, ray_dir, gN, v0, bp, fp);

    its.mesh = &mesh;
    its.sn = sN;
    its.gn = gN;
    its.uv = texcoord;
    its.p = hitpoint;
    its.bp = bp;    // back hit point
    its.fp = fp;    // front hit point;
    its.light_idx = rt_data->light_idx;
    return its;
}

}  // namespace optix