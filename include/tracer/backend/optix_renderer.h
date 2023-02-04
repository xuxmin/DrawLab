#pragma once
#include "tracer/scene.h"
#include "optix/host/texture.h"
#include "optix/host/device_context.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/accel.h"
#include "optix/common/optix_params.h"


namespace optix {

/**
 * A simple Opitx-7 renderer that demonstrates how to set up
 * context, module, programs, pipeline, SBT, etc, and perform
 * a valid launch that renders some pixel.
 *
 */
class OptixRenderer {
public:
    /**
     * Performs all setup, including initializing optix,
     * creates module, pipeline, programs, SBT, etc.
     */
    OptixRenderer(drawlab::Scene* scene);

    /// Render one frame.
    void render();

    /// Resize framebuffer to a given size
    void resize(const int height, const int width);

    /// Download the rendered color buffer
    void downloadPixels(unsigned int h_pixels[]);

    void updateCamera();

protected:
    /// Initialize optix and check for errors.
    void initOptix();

    /// @brief Creates and configures a optix device context
    void createContext();

protected:

    int m_width, m_height;

    DeviceContext* deviceContext;

    // drawlab::OptixAccel* m_optix_accel;
    OptixAccel* m_optix_accel;

    /* our launch parameters, on the host, and the buffer to store
       them on the device */
    LaunchParams launchParams;
    CUDABuffer launchParamsBuffer;

    CUDABuffer colorBuffer;

    drawlab::Scene* m_scene;
};

};  // namespace optix