#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/util.h"

namespace optix {

extern "C" __constant__ Params params;

static __forceinline__ __device__ void
traceRadiance(OptixTraversableHandle handle, float3 ray_origin,
              float3 ray_direction, float tmin, float tmax, RadiancePRD* prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_RADIANCE,  // SBT offset
               RAY_TYPE_COUNT,     // SBT stride
               RAY_TYPE_RADIANCE,  // missSBTIndex
               u0, u1);
}

extern "C" __global__ void __raygen__path() {
    const uint3 idx = optixGetLaunchIndex();
    const int w = params.width;
    const int h = params.height;
    const int subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

    float3 result = make_float3(0.f);
    int i = params.spp;
    do {
        float3 ray_origin, ray_direction;
        params.camera.sampleRay(w, h, idx, seed, ray_origin, ray_direction);

        float eta = 1.f;
        float3 throughput = make_float3(1.f);

        RadiancePRD prd;
        prd.radiance = make_float3(0.f);
        prd.done = false;
        prd.seed = seed;
        prd.sRec = BSDFSampleRecord();

        for (int depth = 0;; ++depth) {
            traceRadiance(params.handle, ray_origin, ray_direction,
                          0.01f,  // tmin
                          1e16f,  // tmax
                          &prd);

            result += prd.radiance * throughput;

            throughput = throughput * prd.sRec.fr;
            eta = eta * prd.sRec.eta;

            if (prd.done) {
                break;
            }

            if (depth > 3) {
                float p = fminf(fmaxf(throughput) * eta * eta, 0.99f);
                if (rnd(seed) > p) {
                    break;
                }
                throughput /= p;
            }

            ray_origin = prd.sRec.p;
            ray_direction = prd.sRec.wo;

            ++depth;
        }
    } while (--i);

    const unsigned int image_index = idx.x + idx.y * params.width;
    float3 accum_color = result / static_cast<float>(params.spp);

    if (subframe_index > 0) {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = params.color_buffer[image_index];
        accum_color = lerp(accum_color_prev, accum_color, a);
    }

    if (invalid_color(accum_color)) {
        printf("Invalid pixel found! idx: (%d, %d), frame_idx {%d}\n", idx.x, idx.y, subframe_index);
    }

    params.color_buffer[image_index] = accum_color;
}
}  // namespace optix