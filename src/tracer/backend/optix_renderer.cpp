#include "tracer/backend/optix_renderer.h"
#include "core/bitmap/bitmap.h"
#include "editor/gui.h"
#include "editor/utils.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
// this include may only appear in a single source file:
#include "stb_image_write.h"
#include <GLFW/glfw3.h>
#include <optix_function_table_definition.h>
#include <spdlog/spdlog.h>

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene, int device_id)
    : m_scene(scene) {
    spdlog::info("[OPTIX RENDERER] Step 1. Initializing optix ...");
    optix::initOptix();

    spdlog::info("[OPTIX RENDERER] Step 2. Creating optix context ...");
    m_device_context = new DeviceContext(device_id);
    m_device_context->configurePipelineOptions();

    spdlog::info("[OPTIX RENDERER] Step 3. Creating raygen programs ...");
    m_device_context->createRaygenProgramsAndBindSBT(
        "optix/integrator/path.cu", "__raygen__path");

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
            return meshs[shape_id]->getBSDF()->getOptixMaterial(
                *m_device_context);
        });

    spdlog::info("[OPTIX RENDERER] Step 7. Setting up optix pipeline ...");
    m_device_context->createPipeline();

    initLaunchParams();
    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    initContext();
}

void OptixRenderer::destroy() {
    delete m_device_context;
    delete m_launch_param;
    delete context;
    delete m_display;
}

OptixRenderer::~OptixRenderer() {
    destroy();
}

void OptixRenderer::initLaunchParams() {
    m_launch_param = new LaunchParam(*m_device_context);

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0], m_height = outputSize[1];
    m_launch_param->setupColorBuffer(m_width, m_height);
    m_launch_param->setupCamera(camera->getOptixCamera());

    std::vector<Light> lights;
    for (const auto emitter : m_scene->getEmitters()) {
        Light light;
        emitter->getOptixLight(light);
        lights.push_back(light);
    }
    m_launch_param->setupLights(lights);

    m_launch_param->setupSampler(m_scene->getSampler()->getSampleCount());
}

void OptixRenderer::initContext() {
    context = new Context();

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0], m_height = outputSize[1];
    float aspect = m_width / m_height;

    context->initCameraState(camera->getOptixCamera(), aspect);
}

void OptixRenderer::renderAsync(std::string filename, bool gui) {
    m_launch_param->updateParamsBuffer();
    m_device_context->launch(*m_launch_param);
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    if (gui) {
        drawlab::GUI gui(m_width, m_height);
        gui.setRenderer(this);
        gui.init();
        gui.start();
    }

    drawlab::Bitmap bitmap(m_height, m_width);
    float3* pixels = (float3*)bitmap.getPtr();
    m_launch_param->getColorData(pixels);

    size_t lastdot = filename.find_last_of(".");
    if (lastdot != std::string::npos)
        filename.erase(lastdot, std::string::npos);

    bitmap.savePNG(filename);
    spdlog::info("Image rendered, and saved to {} ... done.", filename);
}

void OptixRenderer::init() {
    m_display = new opengl::Display(opengl::Display::BufferImageFormat::FLOAT3);
}

void OptixRenderer::render() {
    
    auto t0 = std::chrono::steady_clock::now();

    if (context->camera_changed) {
        m_launch_param->setupCamera(context->getOptixCamera());
    }

    m_launch_param->updateParamsBuffer();

    auto t1 = std::chrono::steady_clock::now();
    context->state_update_time += t1 - t0;
    t0 = t1;

    m_device_context->launch(*m_launch_param);
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
    t1 = std::chrono::steady_clock::now();
    context->render_time += t1 - t0;
    t0 = t1;

    const int w = drawlab::GUI::window_width;
    const int h = drawlab::GUI::window_height;

    drawlab::Bitmap bitmap(m_height, m_width);
    float3* pixels = (float3*)bitmap.getPtr();
    m_launch_param->getColorData(pixels);

    bitmap.flipud();
    bitmap.resize(h, w);
    unsigned int pbo = m_display->getPBO(w, h, bitmap.getPtr());
    m_display->display(w, h, w, h, pbo);

    t1 = std::chrono::steady_clock::now();
    drawlab::displayStats(context->state_update_time, context->render_time,
                          context->display_time);
}

void OptixRenderer::resize(size_t w, size_t h) {
    m_width = (int)w;
    m_height = (int)h;
    m_launch_param->setupColorBuffer(m_width, m_height);
    context->camera.setAspectRatio(m_width / (float)m_height);
}

void OptixRenderer::keyEvent(char key) {}

void OptixRenderer::cursorEvent(float x, float y) {
    if (context->mouse_botton == GLFW_MOUSE_BUTTON_LEFT) {
        context->trackball.setViewMode(Trackball::LookAtFixed);
        context->trackball.updateTracking(
            static_cast<int>(x), static_cast<int>(y),
            drawlab::GUI::window_width, drawlab::GUI::window_height);
        context->camera_changed = true;
    }
    else if (context->mouse_botton == GLFW_MOUSE_BUTTON_RIGHT) {
        context->trackball.setViewMode(Trackball::EyeFixed);
        context->trackball.updateTracking(
            static_cast<int>(x), static_cast<int>(y),
            drawlab::GUI::window_width, drawlab::GUI::window_height);
        context->camera_changed = true;
    }
}

void OptixRenderer::scrollEvent(float offset_x, float offset_y) {
    if (context->trackball.wheelEvent((int)offset_y))
        context->camera_changed = true;
}

void OptixRenderer::mouseButtonEvent(int button, int event, float xpos,
                                     float ypos) {
    if (event == GLFW_PRESS) {
        context->mouse_botton = button;
        context->trackball.startTracking(static_cast<int>(xpos),
                                         static_cast<int>(ypos));
    }
    else {
        context->mouse_botton = -1;
    }
}
}  // namespace optix