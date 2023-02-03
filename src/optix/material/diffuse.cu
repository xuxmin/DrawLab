#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/util.h"


namespace optix {

/**
 * Launch-varying parameters.
 *
 * This params can be accessible from any module in a pipeline.
 * - declare with extern "C" and __constant__
 * - set in OptixPipelineCompileOptions
 * - filled in by optix upon optixLaunch
 */
extern "C" __constant__ LaunchParams params;


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

//---------------------------------------------------------------------
// These program types are specified by prefixing the program’s name with the
// following
//  Ray generation          __raygen__
//  Intersection            __intersection__
//  Any-hit                 __anyhit__
//  Closest-hit             __closesthit__
//  Miss                    __miss__
//  Direct callable         __direct_callable__
//  Continuation callable   __continuation_callable__
//  Exception               __exception__
//
// Each program may call a specific set of device-side intrinsics that
// implement the actual ray-tracing-specific features
//---------------------------------------------------------------------

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const GeometryData::TriangleMesh& mesh_data =
        reinterpret_cast<const GeometryData::TriangleMesh&>(
            rt_data->geometry_data.triangle_mesh);
    const MaterialData& mat_data =
        reinterpret_cast<const MaterialData&>(rt_data->material_data);

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh_data.indices[prim_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------

    const float3 v0 = mesh_data.positions[index.x];
    const float3 v1 = mesh_data.positions[index.y];
    const float3 v2 = mesh_data.positions[index.z];
    float3 geometry_normal = normalize(cross(v1 - v0, v2 - v0));
    float3 shading_normal = geometry_normal;

    if (mesh_data.normals) {
        shading_normal = (1.f - u - v) * mesh_data.normals[index.x] +
                         u * mesh_data.normals[index.y] +
                         v * mesh_data.normals[index.z];
    }

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    geometry_normal = faceforward(geometry_normal, -ray_dir, geometry_normal);
    if (dot(geometry_normal, shading_normal) < 0.f)
        shading_normal -=
            2.f * dot(geometry_normal, shading_normal) * geometry_normal;
    shading_normal = normalize(shading_normal);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    float4 diffuseColor = mat_data.diffuse.albedo;
    if (mat_data.diffuse.albedo_tex) {
        const float2 tc = (1.f - u - v) * mesh_data.texcoords[index.x] +
                          u * mesh_data.texcoords[index.y] +
                          v * mesh_data.texcoords[index.z];

        float4 fromTexture =
            tex2D<float4>(mat_data.diffuse.albedo_tex, tc.x, tc.y);
        diffuseColor *= fromTexture;
    }

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const float3 surfPos = (1.f - u - v) * v0 + u * v1 + v * v2;
    // printf("%lf %lf %lf\n", surfPos.x, surfPos.y, surfPos.z);
    const float3 lightPos = make_float3(-9., 20.f, 0.f);
    const float3 lightDir = lightPos - surfPos;
    const float Ldist = length(lightPos - surfPos);

    // trace shadow ray:
    const bool occluded = traceOcclusion(
        params.handle, surfPos + 1e-3f * geometry_normal, lightDir,
        0.01f,         // tmin
        Ldist - 0.01f  // tmax
    );

    // ------------------------------------------------------------------
    // perform some simple "NdotD" shading
    // ------------------------------------------------------------------

    const float cosDN = 0.2f + .8f * fabsf(dot(ray_dir, shading_normal));
    RadiancePRD* prd = getPRD<RadiancePRD>();

    if (occluded) {
        prd->radiance = make_float3(0.f);
    }
    else {
        prd->radiance = make_float3(cosDN * diffuseColor);
    }
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void
__anyhit__occlusion() { /*! for this simple example, this will remain empty */
}

}  // namespace optix