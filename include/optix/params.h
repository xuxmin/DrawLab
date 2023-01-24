#pragma once

namespace optix {

struct LaunchParams {
    int frameID{0};

    // frame
    unsigned int* color_buffer;
    int width;
    int height;

    // camera
    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;
};

}