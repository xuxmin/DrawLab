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
#include "optix/host/material_table.h"

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene, int device_id)
    : m_scene(scene), m_display(nullptr) {
    spdlog::info("[OPTIX RENDERER] Step 1. Initializing optix ...");
    optix::initOptix();

    spdlog::info("[OPTIX RENDERER] Step 2. Creating optix context ...");
    m_device_context = new DeviceContext(device_id);
    m_device_context->configurePipelineOptions();

    spdlog::info("[OPTIX RENDERER] Step 3. Creating raygen programs ...");
    m_device_context->createRaygenProgramsAndBindSBT(
        "optix/cuda/integrator/path.cu", "__raygen__path");

    spdlog::info("[OPTIX RENDERER] Step 4. Creating miss programs ...");
    m_device_context->createMissProgramsAndBindSBT(
        "optix/cuda/miss.cu", {"__miss__radiance", "__miss__occlusion"});

    spdlog::info("[OPTIX RENDERER] Step 5. Creating optix accel ...");
    m_device_context->createAccel([&](OptixAccel* accel) {
        const auto & meshs = m_scene->getMeshes();
        const auto & light_idx = m_scene->getMeshLightIdx();
        const auto & material_idx = m_scene->getMeshBsdfIdx();
        for (int i = 0; i < meshs.size(); i++) {
            accel->addTriangleMesh(
                meshs[i]->getVertexPosition(), meshs[i]->getVertexIndex(),
                meshs[i]->getVertexNormal(), meshs[i]->getVertexTexCoord(),
                light_idx[i], material_idx[i], meshs[i]->pdfPosition());
        }
    });

    spdlog::info("[OPTIX RENDERER] Step 6. Creating hitgroup programs ...");
    const std::vector<drawlab::Mesh*> meshs = m_scene->getMeshes();
    m_device_context->createHitProgramsAndBindSBT(
        "optix/cuda/hitgroup_pgs.cu",
        {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
        {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}});

    spdlog::info("[OPTIX RENDERER] Step 7. Creating callable programs ...");
    const std::vector<const drawlab::BSDF*> bsdfs = m_scene->getBSDFs();
    std::vector<std::string> cu_files;
    std::vector<std::string> func_names;
    for (auto bsdf : bsdfs) {

        auto type = bsdf->getOptixBSDFType();
        cu_files.push_back(MaterialCUFiles[type]);
        auto funcs = MaterialCallableFuncs[type];
        for (auto func : funcs) {
            func_names.push_back(func);
        }
    }
    m_device_context->createCallableProgramsAndBindSBT(cu_files, func_names);

    spdlog::info("[OPTIX RENDERER] Step 8. Setting up optix pipeline ...");
    m_device_context->createPipeline();

    initLaunchParams();
    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    initContext();
}

void OptixRenderer::destroy() {
    if (m_device_context)
        delete m_device_context;
    if (m_launch_param)
        delete m_launch_param;
    if (m_context)
        delete m_context;
    if (m_display)
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
    m_device_context->getAccel()->packEmittedMesh(lights);
    m_launch_param->setupLights(lights);

    std::vector<Material> mats;
    for (const auto bsdf : m_scene->getBSDFs()) {
        Material mat;
        bsdf->createOptixBSDF(*m_device_context, mat);
        mats.push_back(mat);
    }
    m_launch_param->setupMaterials(mats);

    m_launch_param->setupSampler(m_scene->getSampler()->getSampleCount());
}

void OptixRenderer::initContext() {
    m_context = new Context();

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0], m_height = outputSize[1];
    float aspect = (float)m_width / m_height;

    m_context->initCameraState(camera->getOptixCamera(), aspect);
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

    bitmap.colorNan(drawlab::Color3f(10.f, 0.f, 0.f));
    bitmap.savePNG(filename);
    bitmap.saveEXR(filename);
    spdlog::info("[OPTIX RENDERER] Image rendered, and saved to {} ... done.", filename);
}

void OptixRenderer::init() {
    m_display = new opengl::Display(opengl::Display::BufferImageFormat::FLOAT3);
}

void OptixRenderer::render() {
    
    CUDA_SYNC_CHECK();
    auto t0 = std::chrono::steady_clock::now();

    if (m_context->camera_changed) {
        m_context->camera_changed = false;
        m_launch_param->setupCamera(m_context->getOptixCamera());
        m_launch_param->resetFrameIndex();
    }

    m_launch_param->accFrameIndex();
    m_launch_param->updateParamsBuffer();

    CUDA_SYNC_CHECK();
    auto t1 = std::chrono::steady_clock::now();
    m_context->state_update_time += t1 - t0;
    t0 = t1;

    m_device_context->launch(*m_launch_param);
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
    t1 = std::chrono::steady_clock::now();
    m_context->render_time += t1 - t0;
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
    m_context->display_time += t1 - t0;
    drawlab::displayStats(m_context->state_update_time, m_context->render_time,
                          m_context->display_time);

    static char display_text[128];
    sprintf(display_text,
            "eye    : %5.1f %5.1f %5.1f\n"
            "lookat : %5.1f %5.1f %5.1f\n"
            "up     : %5.1f %5.1f %5.1f\n",
            m_context->camera.eye().x, m_context->camera.eye().y, m_context->camera.eye().z,
            m_context->camera.lookat().x, m_context->camera.lookat().y, m_context->camera.lookat().z,
            m_context->camera.up().x, m_context->camera.up().y, m_context->camera.up().z);
    drawlab::displayText(display_text, 10.f, 10.f);
}

void OptixRenderer::resize(size_t w, size_t h) {
    m_width = (int)w;
    m_height = (int)h;
    m_launch_param->setupColorBuffer(m_width, m_height);
    m_context->camera.setAspectRatio(m_width / (float)m_height);
    m_context->camera_changed = true;
}

void OptixRenderer::keyEvent(char key) {}

void OptixRenderer::cursorEvent(float x, float y) {
    if (m_context->mouse_botton == GLFW_MOUSE_BUTTON_LEFT) {
        m_context->trackball.setViewMode(Trackball::LookAtFixed);
        m_context->trackball.updateTracking(
            static_cast<int>(x), static_cast<int>(y),
            drawlab::GUI::window_width, drawlab::GUI::window_height);
        m_context->camera_changed = true;
    }
    else if (m_context->mouse_botton == GLFW_MOUSE_BUTTON_RIGHT) {
        m_context->trackball.setViewMode(Trackball::EyeFixed);
        m_context->trackball.updateTracking(
            static_cast<int>(x), static_cast<int>(y),
            drawlab::GUI::window_width, drawlab::GUI::window_height);
        m_context->camera_changed = true;
    }
}

void OptixRenderer::scrollEvent(float offset_x, float offset_y) {
    if (m_context->trackball.wheelEvent((int)offset_y))
        m_context->camera_changed = true;
}

void OptixRenderer::mouseButtonEvent(int button, int event, float xpos,
                                     float ypos) {
    if (event == GLFW_PRESS) {
        m_context->mouse_botton = button;
        m_context->trackball.startTracking(static_cast<int>(xpos),
                                         static_cast<int>(ypos));
    }
    else {
        m_context->mouse_botton = -1;
    }
}
}  // namespace optix