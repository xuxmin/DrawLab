#include "tracer/backend/optix_renderer.h"
#include "core/bitmap/bitmap.h"
#include "optix/common/vec_math.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
// this include may only appear in a single source file:
#include "stb_image_write.h"
#include <optix_function_table_definition.h>
#include <spdlog/spdlog.h>

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene) : m_scene(scene) {
    initOptix();

    spdlog::info("[OPTIX RENDERER] Creating optix context ...");
    createContext();
    spdlog::info("[OPTIX RENDERER] Setting up module ...");

    spdlog::info("[OPTIX RENDERER] Creating raygen programs ...");
    deviceContext->createRaygenProgramsAndBindSBT("optix/integrator/simple.cu",
                                                  "__raygen__simple");
    spdlog::info("[OPTIX RENDERER] Creating miss programs ...");
    deviceContext->createMissProgramsAndBindSBT(
        "optix/miss/miss.cu", {"__miss__radiance", "__miss__occlusion"});

    deviceContext->createAccel([&](OptixAccel* accel) {
        for (auto mesh : m_scene->getMeshes()) {
            accel->addTriangleMesh(
                mesh->getVertexPosition(), mesh->getVertexIndex(),
                mesh->getVertexNormal(), mesh->getVertexTexCoord());
        }
    });

    launchParams.handle = deviceContext->getHandle();

    spdlog::info("[OPTIX RENDERER] Building SBT ...");
    const std::vector<drawlab::Mesh*> meshs = m_scene->getMeshes();
    deviceContext->createHitProgramsAndBindSBT(
        meshs.size(), RAY_TYPE_COUNT, [&](int shape_id) {
            return meshs[shape_id]->getBSDF()->getOptixMaterial(*deviceContext);
        });

    spdlog::info("[OPTIX RENDERER] Setting up optix pipeline ...");
    deviceContext->createPipeline();

    launchParamsBuffer.alloc(sizeof(launchParams));
    spdlog::info(
        "[OPTIX RENDERER] Context, module, pipeline, etc, all set up ...");

    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    updateCamera();
}

void OptixRenderer::initOptix() {
    spdlog::info("[OPTIX RENDERER] Initializing optix...");

    // Initialize CUDA for this device on this thread
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error(
            "[OPTIX RENDERER] No CUDA capable devices found!");
    spdlog::info("[OPTIX RENDERER] Found {} CUDA devices", numDevices);

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());
    spdlog::info("[OPTIX RENDERER] Successfully initialized optix... yay!");
}

void OptixRenderer::createContext() {
    const int deviceID = 0;

    deviceContext = new DeviceContext(deviceID);
    deviceContext->configurePipelineOptions();
}

void OptixRenderer::render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.width == 0)
        return;

    launchParamsBuffer.upload(&launchParams, 1);
    launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            deviceContext->getPipeline(),
                            deviceContext->getStream(),
                            /*! parameters and SBT */
                            launchParamsBuffer.devicePtr(),
                            launchParamsBuffer.m_size_in_bytes,
                            &deviceContext->getSBT(),
                            /*! dimensions of the launch: */
                            launchParams.width, launchParams.height, 1));
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

void OptixRenderer::updateCamera() {
    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    resize(outputSize[1], outputSize[0]);

    camera->packLaunchParameters(launchParams);
}

/*! resize frame buffer to given resolution */
void OptixRenderer::resize(const int height, const int width) {
    // if window minimized
    if (height == 0 | width == 0)
        return;

    m_width = width;
    m_height = height;

    // resize our cuda frame buffer
    colorBuffer.resize(height * width * sizeof(unsigned int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.width = width;
    launchParams.height = height;
    launchParams.color_buffer = (unsigned int*)colorBuffer.m_device_ptr;
}

void OptixRenderer::downloadPixels(unsigned int h_pixels[]) {
    colorBuffer.download(h_pixels, launchParams.height * launchParams.width);
}

}  // namespace optix