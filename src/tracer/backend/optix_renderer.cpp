#include "tracer/backend/optix_renderer.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
#include "core/bitmap/bitmap.h"
#include "editor/gui.h"
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

    m_launch_param = new LaunchParam(*m_device_context);
    spdlog::info(
        "[OPTIX RENDERER] Context, module, pipeline, etc, all set up ...");

    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    updateLaunchParams();
}

void OptixRenderer::renderFrame() {
    m_launch_param->updateParamsBuffer();
    m_device_context->launch(*m_launch_param);
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}


void OptixRenderer::render(std::string filename, const bool gui) {

    renderFrame();

    drawlab::Bitmap bitmap(m_height, m_width);
    float3* pixels = (float3*)bitmap.getPtr();
    m_launch_param->getColorData(pixels);
    drawlab::Bitmap bitmap_copy(bitmap);

    if (gui) {
        bitmap.flipud();
        drawlab::GUI gui(&bitmap);
        gui.init();
        gui.start();
    }

    bitmap_copy.savePNG(filename);
    spdlog::info("Image rendered, and saved to {} ... done.", filename);
}

void OptixRenderer::updateLaunchParams() {

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0];
    m_height = outputSize[1];
    m_launch_param->setupColorBuffer(m_width, m_height);
    m_launch_param->setupCamera(camera->getOptixCamera());


    std::vector<Light> lights;
    for (const auto emitter : m_scene->getEmitters()) {
        Light light;
        emitter->getOptixLight(light);
        lights.push_back(light);
    }
    m_launch_param->setupLights(lights);
}

/*! resize frame buffer to given resolution */
void OptixRenderer::resize(const int height, const int width) {
}

}  // namespace optix