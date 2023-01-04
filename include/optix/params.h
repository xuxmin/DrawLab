#pragma once

namespace optix {

struct LaunchParams {
    int frameID{0};
    float* colorBuffer;
    int frame_width;
    int frame_height;
};

}