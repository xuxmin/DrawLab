#pragma once

#include "optix/host/cuda_buffer.h"
#include "optix/host/cuda_texture.h"
#include "optix/host/device_context.h"
#include "optix/host/function_table.h"
#include "optix/host/optix_accel.h"
#include "optix/host/param_buffer.h"
#include "optix/optix_params.h"

namespace optix {

/**
 * OptixScene object holds all information on scene objects.
 */
class OptixScene {
public:
    OptixScene();

    ~OptixScene();

    // Add light to scene, the order is important.
    void addLight(const Light& light);

    // Add material to scene, the order is important.
    void addMaterial(const Material& material);

    // Update material parameters
    void updateMaterial(int mat_id, bool is_hide);

    // Record textures resource which are link to materials
    void recordTextures(const std::vector<const CUDATexture*>& textures);

    // Setup camera
    void updateCamera(const Camera& camera);

    // Resize frame buffer
    void resize(int width, int height);

    //
    void updateSampler(int spp);

    void updateIntegrator(const Integrator& integrator);

    // Each
    void addMesh(const std::vector<float>& positions,
                 const std::vector<unsigned int>& indices,
                 const std::vector<float>& normals,
                 const std::vector<float>& tangents,
                 const std::vector<float>& texcoords, int light_idx,
                 int material_idx, float pdf);

    void activate();

    void render(bool sync_check = true);

    ParamBuffer* getParamBuffer() { return m_param_buffer; }

private:
    std::vector<Light>                  m_lights;
    std::vector<Material>               m_materials;
    std::vector<OptixSceneMesh>         m_meshs;
    std::vector<const CUDATexture*>     m_textures;

    Integrator  m_integrator;
    Camera      m_camera;
    int         m_width;
    int         m_height;
    int         m_spp;

    DeviceContext*  m_device_context;
    OptixAccel*     m_optix_accel;
    ParamBuffer*    m_param_buffer;

    bool        m_activated;
};

}  // namespace optix