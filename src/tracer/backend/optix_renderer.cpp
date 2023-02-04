#include "tracer/backend/optix_renderer.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include "stb_image_write.h"
#include <spdlog/spdlog.h>

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene, int device_id) : m_scene(scene) {

    spdlog::info("[OPTIX RENDERER] Step 1. Initializing optix ...");
    optix::initOptix();

    spdlog::info("[OPTIX RENDERER] Step 2. Creating optix context ...");
    m_device_context = new DeviceContext(device_id);
    m_device_context->configurePipelineOptions();

    spdlog::info("[OPTIX RENDERER] Step 3. Creating raygen programs ...");
    m_device_context->createRaygenProgramsAndBindSBT(
        "optix/integrator/simple.cu", "__raygen__simple");

    spdlog::info("[OPTIX RENDERER] Step 4. Creating miss programs ...");
    m_device_context->createMissProgramsAndBindSBT(
        "optix/miss/miss.cu", {"__miss__radiance", "__miss__occlusion"});

    spdlog::info("[OPTIX RENDERER] Step 5. Creating optix accel ...");
    m_device_context->createAccel([&](OptixAccel* accel) {
        for (auto mesh : m_scene->getMeshes()) {
            accel->addTriangleMesh(
                mesh->getVertexPosition(), mesh->getVertexIndex(),
                mesh->getVertexNormal(), mesh->getVertexTexCoord());
        }
    });

    spdlog::info("[OPTIX RENDERER] Step 6. Creating hitgroup programs ...");
    const std::vector<drawlab::Mesh*> meshs = m_scene->getMeshes();
    m_device_context->createHitProgramsAndBindSBT(
        meshs.size(), RAY_TYPE_COUNT, [&](int shape_id) {
            return meshs[shape_id]->getBSDF()->getOptixMaterial(*m_device_context);
        });

    spdlog::info("[OPTIX RENDERER] Step 7. Setting up optix pipeline ...");
    m_device_context->createPipeline();

    m_launch_params_buffer.alloc(sizeof(m_launch_params));
    spdlog::info(
        "[OPTIX RENDERER] Context, module, pipeline, etc, all set up ...");

    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    updateLaunchParams();
}

void OptixRenderer::render() {
    m_launch_params_buffer.upload(&m_launch_params, 1);
    m_launch_params.frameID++;

    m_device_context->launch(m_launch_params_buffer, m_launch_params.width,
                             m_launch_params.height);

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    // write image
    std::vector<unsigned int> pixels(m_width * m_height);
    downloadPixels(pixels.data());
    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(), m_width, m_height, 4, pixels.data(),
                   m_width * sizeof(unsigned int));
    spdlog::info("Image rendered, and saved to {} ... done.", fileName);
}

void OptixRenderer::updateLaunchParams() {
    m_launch_params.handle = m_device_context->getHandle();

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    resize(outputSize[1], outputSize[0]);

    camera->packLaunchParameters(m_launch_params);
}

/*! resize frame buffer to given resolution */
void OptixRenderer::resize(const int height, const int width) {
    m_width = width;
    m_height = height;

    // resize our cuda frame buffer
    m_color_buffer.resize(height * width * sizeof(unsigned int));

    // update the launch parameters that we'll pass to the optix
    // launch
    m_launch_params.width = width;
    m_launch_params.height = height;
    m_launch_params.color_buffer = (unsigned int*)m_color_buffer.m_device_ptr;
}

void OptixRenderer::downloadPixels(unsigned int h_pixels[]) {
    m_color_buffer.download(h_pixels, m_launch_params.height * m_launch_params.width);
}

}  // namespace optix