#pragma once

#include "optix/host/cuda_buffer.h"
#include "optix/host/cuda_texture.h"
#include "optix/optix_params.h"

namespace optix {

class ParamBuffer {
public:
    ParamBuffer();

    ~ParamBuffer();

    void updateColorBuffer(int width, int height);

    void updateLightBuffer(const std::vector<Light>& lights);

    void updateMaterialBuffer(const std::vector<Material>& materials);

    void updateCamera(const Camera& camera);

    void updateSampler(const int spp);

    void updateSceneHandle(OptixTraversableHandle handle);

    void updateParamBuffer();

    void resetFrameIndex();

    void accFrameIndex();

    void getColorData(float3* pixels) const;

    const CUDABuffer& getParamsBuffer() const { return m_params_buffer; }

    int getWidth() const { return m_params.width; }

    int getHeight() const { return m_params.height; }

private:
    Params m_params;
    CUDABuffer m_params_buffer;

    CUDABuffer m_color_buffer;
    CUDABuffer m_light_buffer;
    CUDABuffer m_material_buffer;

    bool m_dirty;
};

}  // namespace optix