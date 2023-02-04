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
    void render();

    /// Resize framebuffer to a given size
    void resize(const int height, const int width);

    /// Download the rendered color buffer
    void downloadPixels(unsigned int h_pixels[]);

    void updateLaunchParams();

protected:
    int m_width, m_height;

    DeviceContext* m_device_context;

    LaunchParams m_launch_params;
    CUDABuffer m_launch_params_buffer;

    CUDABuffer m_color_buffer;

    drawlab::Scene* m_scene;
};

};  // namespace optix