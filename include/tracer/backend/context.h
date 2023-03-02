#pragma once

#include "core/math/vector.h"
#include "optix/camera/camera.h"
#include "optix/host/trackball.h"
#include <chrono>

namespace optix {

struct Context {
    bool camera_changed = false;
    bool hide_light = false;
    float3 bg_color;
    float epsilon = 1e-3;
    optix::PerspectiveCamera camera;
    optix::Trackball trackball;

    int mouse_botton = -1;

    void initCameraState(const optix::Camera& cam, float aspect);

    Camera getOptixCamera() const;

    std::chrono::duration<double> state_update_time;
    std::chrono::duration<double> render_time;
    std::chrono::duration<double> display_time;
};

}  // namespace optix