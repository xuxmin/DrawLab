#include "optix/host/optix_scene.h"
#include <spdlog/spdlog.h>

namespace optix {

OptixScene::OptixScene() {
    optix::initOptix();
    m_device_context = new DeviceContext(0);
    m_param_buffer = new ParamBuffer();
    m_optix_accel = nullptr;
    m_activated = false;
    m_spp = 1;
    m_width = 0;
    m_height = 0;
}

void OptixScene::addLight(const Light& light) {
    if (m_activated) {
        spdlog::error("[OPTIX SCENE] Can't add light after optix scene activate.");
    }
    m_lights.push_back(light); 
}

void OptixScene::addMaterial(const Material& material) {
    if (m_activated) {
        spdlog::error("[OPTIX SCENE] Can't add material after optix scene activate.");
    }
    m_materials.push_back(material);
}

void OptixScene::updateMaterial(int mat_id, bool is_hide) {
    m_materials[mat_id].is_hide = is_hide;
    m_param_buffer->updateMaterialBuffer(m_materials);
}

void OptixScene::recordTextures(const std::vector<const CUDATexture*>& textures) {
    if (m_textures.size() > 0) {
        spdlog::error("[OPTIX SCENE] OptixScene::recordTextures() can be called only once");
    }
    m_textures = textures;
}

void OptixScene::updateCamera(const Camera& camera) {
    m_camera = camera;
    m_param_buffer->updateCamera(camera);
}

void OptixScene::resize(int width, int height) {
    m_width = width;
    m_height = height;
    m_param_buffer->updateColorBuffer(width, height);
}

void OptixScene::updateSampler(int spp) { 
    m_spp = spp;
    m_param_buffer->updateSampler(spp);
}

void OptixScene::updateIntegrator(const Integrator& integrator) {
    m_integrator = integrator;
}

void OptixScene::addMesh(const std::vector<float>& positions,
                         const std::vector<unsigned int>& indices,
                         const std::vector<float>& normals,
                         const std::vector<float>& tangents,
                         const std::vector<float>& texcoords, int light_idx,
                         int material_idx, float pdf) {
    m_meshs.push_back(OptixSceneMesh(positions, indices, normals, tangents,
                                     texcoords, light_idx, material_idx, pdf));
}

void OptixScene::activate() {
    m_activated = true;

    spdlog::info("[OPTIX SCENE] Start activate optix scene...");
    m_device_context->configurePipelineOptions();

    spdlog::info("[OPTIX SCENE] Step 1. Creating raygen programs ...");
    m_device_context->buildRaygenProgramsAndBindSBT(
        IntegratorTables[m_integrator.type][0],
        IntegratorTables[m_integrator.type][1]);

    spdlog::info("[OPTIX SCENE] Step 2. Creating miss programs ...");
    m_device_context->buildMissProgramsAndBindSBT(
        "optix/cuda/miss.cu", {"__miss__radiance", "__miss__occlusion"});

    spdlog::info("[OPTIX SCENE] Step 3. Creating optix accel ...");
    m_optix_accel = new OptixAccel(m_device_context->getOptixDeviceContext());
    m_optix_accel->build(m_meshs);

    spdlog::info("[OPTIX SCENE] Step 4. Creating hitgroup programs ...");
    m_device_context->buildHitProgramsAndBindSBT(
        "optix/cuda/hitgroup_pgs.cu",
        {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
        {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}}, m_optix_accel);

    spdlog::info("[OPTIX SCENE] Step 5. Creating callable programs ...");
    std::vector<std::string> cu_files;
    std::vector<std::string> func_names;
    for (auto mat : m_materials) {
        cu_files.push_back(MaterialCUFiles[mat.type]);
        for (auto func : MaterialCallableFuncs[mat.type]) {
            func_names.push_back(func);
        }
    }
    m_device_context->buildCallableProgramsAndBindSBT(cu_files, func_names);

    spdlog::info("[OPTIX RENDERER] Step 6. Setting up optix pipeline ...");
    m_device_context->buildPipeline();

    // Init launch params
    m_param_buffer->updateSceneHandle(m_optix_accel->getHandle());
    m_param_buffer->updateColorBuffer(m_width, m_height);
    m_param_buffer->updateCamera(m_camera);

    m_optix_accel->packEmittedMesh(m_lights);
    m_param_buffer->updateLightBuffer(m_lights);
    m_param_buffer->updateMaterialBuffer(m_materials);
    m_param_buffer->updateIntegrator(m_integrator);
    m_param_buffer->updateSampler(m_spp);
    m_param_buffer->updateParamBuffer();
}

void OptixScene::render(bool sync_check) {
    if (!m_activated) {
        spdlog::warn("[OPTIX SCENE]: Please activate the optix scene before rendering.");
        return;
    }
    if (m_width == 0 || m_height == 0) {
        spdlog::warn("[OPTIX SCENE]: The width or height is zero!");
        return;
    }
    m_param_buffer->accFrameIndex();
    m_param_buffer->updateParamBuffer();
    m_device_context->launch(m_param_buffer);

    if (sync_check)
        CUDA_SYNC_CHECK();
}

OptixScene::~OptixScene() {
    for (auto tex : m_textures) {
        delete tex;
    }

    delete m_device_context;
    delete m_optix_accel;
    delete m_param_buffer;
}

}  // namespace optix
