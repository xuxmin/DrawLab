#pragma once
#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/device_context.h"
#include "tracer/scene.h"

namespace optix {

class OptixRenderer {
public:
    OptixRenderer(drawlab::Scene* scene, int device_id = 0);

    /// Render one frame.
    void render(std::string filename, const bool gui = false);

    /// Resize framebuffer to a given size
    void resize(const int height, const int width);

    void updateLaunchParams();

private:
    void renderFrame();

protected:
    int m_width, m_height;

    DeviceContext* m_device_context;

    LaunchParams m_launch_params;
    CUDABuffer m_launch_params_buffer;

    CUDABuffer m_color_buffer;
    CUDABuffer m_light_buffer;

    drawlab::Scene* m_scene;
};

};  // namespace optix