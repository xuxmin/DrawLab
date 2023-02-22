#include "tracer/backend/optix_renderer.h"
#include "core/bitmap/bitmap.h"
#include "editor/gui.h"
#include "editor/utils.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
#include "stb_image_write.h"
#include <GLFW/glfw3.h>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <spdlog/spdlog.h>
#include <imgui/imgui.h>

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene, int device_id)
    : m_display(nullptr) {
    
    m_scene = scene;
    m_context = initContext(scene);
    m_optix_scene = initOptixScene(scene);
    m_optix_scene->activate();
}

OptixScene* OptixRenderer::initOptixScene(drawlab::Scene* scene) {
    m_optix_scene = new OptixScene();

    // Add lights to scene
    for (auto emitter : scene->getEmitters()) {
        Light light;
        emitter->getOptixLight(light);
        m_optix_scene->addLight(light);
    }

    // Add material to scene
    std::vector<const CUDATexture*> texs;
    for (auto bsdf : scene->getBSDFs()) {
        Material mat;
        bsdf->createOptixMaterial(mat, texs);
        m_optix_scene->addMaterial(mat);
    }
    m_optix_scene->recordTextures(texs);

    // Update materials, hide lights
    for (auto bsdf_idx : m_scene->getLightBsdfIdx()) {
        m_optix_scene->updateMaterial(bsdf_idx, m_context->hide_light);
    }

    // Update bg color
    m_optix_scene->getParamBuffer()->updateBgColor(m_context->bg_color);

    // Update envmap
    m_optix_scene->getParamBuffer()->updateEnvMap(m_scene->getEnvironmentEmitterIdx());

    // Add shape to scene
    const auto & meshs = scene->getMeshes();
    const auto & light_idx = scene->getMeshLightIdx();
    const auto & material_idx = scene->getMeshBsdfIdx();
    for (int i = 0; i < meshs.size(); i++) {
        m_optix_scene->addMesh(
            meshs[i]->getVertexPosition(), meshs[i]->getVertexIndex(),
            meshs[i]->getVertexNormal(), meshs[i]->getVertexTangent(),
            meshs[i]->getVertexTexCoord(), light_idx[i], material_idx[i],
            meshs[i]->pdfPosition());
    }

    // Config camera
    const drawlab::Camera* camera = scene->getCamera();
    m_optix_scene->updateCamera(camera->getOptixCamera());

    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0], m_height = outputSize[1];
    m_optix_scene->resize(m_width, m_height);

    // Config integrator
    m_optix_scene->updateIntegrator(scene->getIntegrator()->getOptixIntegrator());

    // Config sampler
    m_optix_scene->updateSampler(scene->getSampler()->getSampleCount());

    return m_optix_scene;
}

void OptixRenderer::updateOptixScene() {
    // Update camera
    if (m_context->camera_changed) {
        m_context->camera_changed = false;
        m_optix_scene->updateCamera(m_context->getOptixCamera());
        m_optix_scene->getParamBuffer()->resetFrameIndex();
    }

    // Update materials, hide lights
    for (auto bsdf_idx : m_scene->getLightBsdfIdx()) {
        m_optix_scene->updateMaterial(bsdf_idx, m_context->hide_light);
    }
    
    // update bg color
    m_optix_scene->getParamBuffer()->updateBgColor(m_context->bg_color);

    // update epsilon
    m_optix_scene->getParamBuffer()->updateSceneEpsilon(m_context->epsilon);
}

OptixRenderer::~OptixRenderer() {
    if (m_context)
        delete m_context;
    if (m_display)
        delete m_display;
}

Context* OptixRenderer::initContext(drawlab::Scene* scene) {
    Context* context = new Context();

    const drawlab::Camera* camera = scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    m_width = outputSize[0], m_height = outputSize[1];
    float aspect = (float)m_width / m_height;

    context->initCameraState(camera->getOptixCamera(), aspect);
    auto bg_color = scene->getBgColor();
    context->bg_color = make_float3(bg_color[0], bg_color[1], bg_color[2]);
    return context;
}

void OptixRenderer::renderAsync(std::string filename, bool gui) {
    m_optix_scene->render();

    if (gui) {
        drawlab::GUI gui(m_width, m_height);
        gui.setRenderer(this);
        gui.init();
        gui.start();
    }

    drawlab::Bitmap bitmap(m_height, m_width);
    float3* pixels = (float3*)bitmap.getPtr();
    m_optix_scene->getParamBuffer()->getColorData(pixels);

    size_t lastdot = filename.find_last_of(".");
    if (lastdot != std::string::npos)
        filename.erase(lastdot, std::string::npos);

    // bitmap.colorNan(drawlab::Color3f(10.f, 0.f, 0.f));
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

    updateOptixScene();

    CUDA_SYNC_CHECK();
    auto t1 = std::chrono::steady_clock::now();
    m_context->state_update_time += t1 - t0;
    t0 = t1;

    m_optix_scene->render();

    t1 = std::chrono::steady_clock::now();
    m_context->render_time += t1 - t0;
    t0 = t1;

    const int w = drawlab::GUI::window_width;
    const int h = drawlab::GUI::window_height;

    drawlab::Bitmap bitmap(m_height, m_width);
    float3* pixels = (float3*)bitmap.getPtr();
    m_optix_scene->getParamBuffer()->getColorData(pixels);

    bitmap.flipud();
    bitmap.resize(h, w);
    unsigned int pbo = m_display->getPBO(w, h, bitmap.getPtr());
    m_display->display(w, h, w, h, pbo);

    t1 = std::chrono::steady_clock::now();
    m_context->display_time += t1 - t0;

    // ImGUI
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("TextOverlayFG", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize);
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
    ImGui::Checkbox("Hide Lights", &m_context->hide_light);
    ImGui::ColorEdit3("Background Color", (float*)&m_context->bg_color);
    ImGui::DragFloat("Scene Epsilon", &m_context->epsilon, 1e-4, 1e-4, 1e-1, "%.4f");
    ImGui::End();
}

void OptixRenderer::resize(size_t w, size_t h) {
    m_width = (int)w;
    m_height = (int)h;
    m_optix_scene->resize(m_width, m_height);
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