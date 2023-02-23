#include "optix/math/vec_math.h"
#include "shader_common.h"
#include "per_ray_data.h"

namespace optix {

extern "C" __constant__ Params params;

#define NEXT_EVENT_ESTIMATION

extern "C" __global__ void __miss__radiance() {

    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_ori = optixGetWorldRayOrigin();
    RadiancePRD* prd = getPRD<RadiancePRD>();

    prd->radiance = make_float3(0.f, 0.f, 0.f);

    const int env_idx = params.envmap_idx;
    if (env_idx >= 0) {

        const Light& light = params.light_buffer.lights[env_idx];
        const BSDFSampleRecord& sRec = prd->sRec;
        float3 light_val = light.eval(Intersection(), ray_dir);
        // Set background
        if (prd->depth == 0) {
            if (light.envmap.visual) {
                prd->radiance = light_val;
            }
            else {
                prd->radiance = params.bg_color;
            }
        }
        else {  // Set indirect light
            float mis = 1.f;

#ifdef NEXT_EVENT_ESTIMATION
            LightSampleRecord dRec;
            dRec.d = ray_dir;
            float light_pdf = params.light_buffer.pdfLightDirection(env_idx, dRec);
            float bsdf_pdf = sRec.pdf;
            mis = sRec.is_diffuse ? powerHeuristic(bsdf_pdf, light_pdf) : 1.f;
#endif
            prd->radiance = mis * light_val;
        }
    }

    prd->done = true;
}

extern "C" __global__ void __miss__occlusion() {
    // setPayloadOcclusion(true);
}

}  // namespace optix