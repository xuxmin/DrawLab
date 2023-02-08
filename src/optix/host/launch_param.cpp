#include "optix/host/launch_param.h"
#include "optix/host/device_context.h"

namespace optix {

LaunchParam::LaunchParam(const DeviceContext& device_context)
    : m_device_context(device_context), dirty(false) {
    m_params_buffer.alloc(sizeof(Params));
}

LaunchParam::~LaunchParam() {
    m_params_buffer.free();
    m_color_buffer.free();
    m_light_buffer.free();
}

void LaunchParam::setupColorBuffer(int width, int height) {
    m_width = width;
    m_height = height;

    // resize our cuda frame buffer
    m_color_buffer.resize(height * width * sizeof(float3));

    dirty = true;
}

void LaunchParam::setupLights(const std::vector<Light>& lights) {
    m_light_num = lights.size();
    m_light_buffer.allocAndUpload(lights);

    dirty = true;
}

void LaunchParam::setupCamera(const Camera& camera) {
    m_camera = camera;

    dirty = true;
}

void LaunchParam::setupSampler(const int spp) {
    m_params.spp = spp;
    dirty = true;
}

void LaunchParam::updateParamsBuffer() {
    if (!dirty) {
        return;
    }
    // color buffer
    m_params.width = m_width;
    m_params.height = m_height;
    m_params.color_buffer = (float3*)m_color_buffer.m_device_ptr;

    // light buffer
    m_params.light_num = m_light_num;
    m_params.lights = (Light*)m_light_buffer.m_device_ptr;

    // camera
    m_params.camera = m_camera;

    // blas
    m_params.handle = m_device_context.getHandle();

    m_params_buffer.upload(&m_params, 1);

    dirty = false;
}

void LaunchParam::getColorData(float3* pixels) const {
    m_color_buffer.download(pixels, m_width * m_height);
}

}  // namespace optix