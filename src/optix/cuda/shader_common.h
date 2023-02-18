#pragma once

#include "optix/optix_params.h"
#include "optix/math/onb.h"
#include <cuda_runtime.h>
#include <optix.h>

namespace optix {

__forceinline__ __device__ bool invalid_color(float3 color) {
    return isnan(color.x) || isnan(color.y) || isnan(color.z);
}

__forceinline__ __host__ __device__ float powerHeuristic(const float a,
                                                         const float b) {
    const float t = a * a;
    return t / (t + b * b);
}

static __forceinline__ __device__ Intersection getHitData(const Material& mat_data) {
    Intersection its;
    // ------------------------------------------------------------------
    // Gather basic hit information
    // ------------------------------------------------------------------

    const HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const TriangleMesh& mesh = reinterpret_cast<const TriangleMesh&>(
        rt_data->geometry_data.triangle_mesh);
    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh.indices[prim_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // Compute texcoord
    // ------------------------------------------------------------------
    float2 texcoord = make_float2(0.f);
    if (mesh.texcoords) {
        const float2 t0 = mesh.texcoords[index.x];
        const float2 t1 = mesh.texcoords[index.y];
        const float2 t2 = mesh.texcoords[index.z];
        texcoord = (1.f - u - v) * t0 + u * t1 + v * t2;
    }

    // ------------------------------------------------------------------
    // Compute geometry normal
    // ------------------------------------------------------------------
    const float3 v0 = mesh.positions[index.x];
    const float3 v1 = mesh.positions[index.y];
    const float3 v2 = mesh.positions[index.z];
    float3 gN = normalize(cross(v1 - v0, v2 - v0));  // geometry normal

    // ------------------------------------------------------------------
    // Compute shading normal
    // ------------------------------------------------------------------
    // default: shading normal is geometry normal
    float3 sN = gN;                                  // shading normal

    // if normal texture is available
    if (mat_data.normal_tex) {
        if (mat_data.normal_in_world) {
            sN = make_float3(tex2D<float4>(mat_data.normal_tex, texcoord.x, texcoord.y));
            sN = sN * 2.f - 1.f;
        }
        else {
            if (mesh.tangents) {
                float3 local_n = make_float3(tex2D<float4>(mat_data.normal_tex, texcoord.x, texcoord.y));
                local_n = local_n * 2.f - 1.f;

                const float3 t0 = mesh.tangents[index.x];
                const float3 t1 = mesh.tangents[index.y];
                const float3 t2 = mesh.tangents[index.z];
                const float3 sT = (1.f - u - v) * t0 + u * t1 + v * t2;
                Onb onb(gN, sT);
                sN = onb.inverse_transform(local_n);
            }
            else {
                printf("The mesh should have tangent, if you want to use a tangent space normal texture.");
            }
        }
    }
    // else consider vertex normal
    else if (mesh.normals) {  
        const float3 n0 = mesh.normals[index.x];
        const float3 n1 = mesh.normals[index.y];
        const float3 n2 = mesh.normals[index.z];
        sN = (1.f - u - v) * n0 + u * n1 + v * n2;
    }
    sN = normalize(sN);

    // ------------------------------------------------------------------
    // Compute shading tangent (for anisotropic material)
    // ------------------------------------------------------------------
    float3 sT = make_float3(0.f);
    if (mat_data.tangent_tex) {
        if (mat_data.tangent_in_world) {
            sT = make_float3(tex2D<float4>(mat_data.tangent_tex, texcoord.x, texcoord.y));
            sT = sT * 2.f - 1.f;
        }
        else {
            if (mesh.tangents) {
                float3 local_t = make_float3(tex2D<float4>(mat_data.tangent_tex, texcoord.x, texcoord.y));
                local_t = local_t * 2.f - 1.f;

                const float3 t0 = mesh.tangents[index.x];
                const float3 t1 = mesh.tangents[index.y];
                const float3 t2 = mesh.tangents[index.z];
                const float3 gT = (1.f - u - v) * t0 + u * t1 + v * t2;
                Onb onb(gN, gT);
                sT = onb.inverse_transform(local_t);
            }
            else {
                printf("The mesh should have tangent, if you want to use a tangent space tangent texture.");
            }
        }
        sT = normalize(sT);
    }

    const float3 hitpoint =
        optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    its.mesh = &mesh;
    its.sn = sN;
    its.gn = gN;
    its.st = sT;
    its.uv = texcoord;
    its.p = hitpoint;
    its.light_idx = rt_data->light_idx;
    return its;
}

static __forceinline__ __device__ float fresnel(float cosThetaI, float extIOR,
                                                float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        float t = etaI;
        etaI = etaT;
        etaT = t;
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta * eta * (1 - cosThetaI * cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f; /* Total internal reflection! */

    float cosThetaT = sqrtf(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT) /
               (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT) /
               (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}

static __forceinline__ __device__ float3 refract(const float3& wi, float eta) {
    float cosThetaI = wi.z;
    bool outside = cosThetaI > 0.f;
    eta = outside ? eta : 1 / eta;
    float cosThetaTSquare = 1 - eta * eta * (1 - cosThetaI * cosThetaI);
    float cosThetaT = sqrtf(cosThetaTSquare);
    cosThetaT = outside ? -cosThetaT : cosThetaT;
    return make_float3(-eta * wi.x, -eta * wi.y, cosThetaT);
}

static __forceinline__ __device__ float3 reflect(const float3& wi, const float3 n) {
    return 2 * dot(wi, n) * n - wi;
}

}  // namespace optix