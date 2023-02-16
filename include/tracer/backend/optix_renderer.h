#pragma once
#include "optix/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/device_context.h"
#include "optix/host/optix_scene.h"
#include "opengl/display.h"
#include "editor/renderer.h"
#include "tracer/backend/context.h"
#include "tracer/scene.h"
#include <chrono>


namespace optix {

class OptixRenderer : public drawlab::Renderer {
public:
    OptixRenderer(drawlab::Scene* scene, int device_id = 0);
    ~OptixRenderer();
    void destroy();

    void init();
    void render();
    void resize(size_t w, size_t h);
    void keyEvent(char key);
    void cursorEvent(float x, float y);
    void scrollEvent(float offset_x, float offset_y);
    void mouseButtonEvent(int button, int event, float xpos, float ypos);

    void renderAsync(std::string filename, bool gui = false);

protected:
    opengl::Display* m_display;
    int m_width, m_height;

    OptixScene* m_optix_scene;
    Context* m_context;

private:
    OptixScene* initOptixScene(drawlab::Scene* scene);
    Context* initContext(drawlab::Scene* scene);
};

};  // namespace optix