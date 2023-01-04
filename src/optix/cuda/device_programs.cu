#include <optix_device.h>

#include "optix/params.h"

using namespace optix;

namespace optix {

/**
 * Launch-varying parameters.
 * 
 * This params can be accessible from any module in a pipeline.
 * - declare with extern "C" and __constant__
 * - set in OptixPipelineCompileOptions
 * - filled in by optix upon optixLaunch
*/
extern "C"  __constant__ LaunchParams optixLaunchParams;

//---------------------------------------------------------------------
// These program types are specified by prefixing the program’s name with the following
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

extern "C" __global__ void
__closesthit__radiance() { /*! for this simple example, this will remain empty
                            */
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void
__miss__radiance() { /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    if (optixLaunchParams.frameID == 0 && optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) {
        // we could of course also have used optixGetLaunchDims to query
        // the launch size, but accessing the optixLaunchParams here
        // makes sure they're not getting optimized away (because
        // otherwise they'd not get used)
        printf("############################################\n");
        printf("Hello world from OptiX 7 raygen program!\n(within a "
               "%ix%i-sized launch)\n",
               optixLaunchParams.frame_width, optixLaunchParams.frame_height);
        printf("############################################\n");
    }

    // ------------------------------------------------------------------
    // for this example, produce a simple test pattern:
    // ------------------------------------------------------------------

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int r = (ix % 256);
    const int g = (iy % 256);
    const int b = ((ix + iy) % 256);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const float rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const unsigned int fbIndex = ix + iy * optixLaunchParams.frame_width;
    optixLaunchParams.colorBuffer[fbIndex] = rgba;
}
}  // namespace optix