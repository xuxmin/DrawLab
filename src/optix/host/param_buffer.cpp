#include "optix/host/param_buffer.h"
#include "optix/host/device_context.h"

namespace optix {

ParamBuffer::ParamBuffer() {
    m_dirty = true;
    m_params.subframe_index = 0;
    m_params_buffer.alloc(sizeof(Params));
}

ParamBuffer::~ParamBuffer() {
    m_params_buffer.free();
    m_color_buffer.free();
    m_light_buffer.free();
}

void ParamBuffer::updateColorBuffer(int width, int height) {
    m_dirty = true;
    m_params.width = width;
    m_params.height = height;
    // resize our cuda frame buffer
    m_color_buffer.resize(height * width * sizeof(float3));
    m_params.color_buffer = (float3*)m_color_buffer.m_device_ptr;
}

void ParamBuffer::updateLightBuffer(const std::vector<Light>& lights) {
    m_dirty = true;
    m_params.light_buffer.light_num = lights.size();
    m_light_buffer.allocAndUpload(lights);
    m_params.light_buffer.lights = (Light*)m_light_buffer.m_device_ptr;
}

void ParamBuffer::updateMaterialBuffer(const std::vector<Material>& materials) {
    m_dirty = true;
    m_material_buffer.allocAndUpload(materials);
    m_params.material_buffer.material_num = materials.size();
    m_params.material_buffer.materials =
        (Material*)m_material_buffer.m_device_ptr;
}

void ParamBuffer::updateCamera(const Camera& camera) {
    m_dirty = true;
    m_params.camera = camera;
}

void ParamBuffer::updateSampler(const int spp) {
    m_dirty = true;
    m_params.spp = spp;
}

void ParamBuffer::updateSceneHandle(OptixTraversableHandle handle) {
    m_params.handle = handle;
}

void ParamBuffer::resetFrameIndex() {
    m_dirty = true;
    m_params.subframe_index = 0;
}

void ParamBuffer::accFrameIndex() {
    m_dirty = true;
    m_params.subframe_index++;
}

void ParamBuffer::updateParamBuffer() {
    if (!m_dirty) {
        return;
    }
    m_params_buffer.upload(&m_params, 1);
    m_dirty = false;
}

void ParamBuffer::getColorData(float3* pixels) const {
    m_color_buffer.download(pixels, m_params.width * m_params.height);
}

}  // namespace optix