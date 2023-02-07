#pragma once
#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/device_context.h"
#include "optix/host/launch_param.h"
#include "opengl/display.h"
#include "editor/renderer.h"
#include "tracer/scene.h"

namespace optix {

class OptixRenderer : public drawlab::Renderer {
public:
    OptixRenderer(drawlab::Scene* scene, int device_id = 0);

    void init();
    void render();
    void resize(size_t w, size_t h);
    void keyEvent(char key);
    void cursorEvent(float x, float y, unsigned char keys);
    void scrollEvent(float offset_x, float offset_y);
    void mouseButtonEvent(int button, int event);

    void renderAsync(std::string filename, bool gui = false);

protected:
    drawlab::Scene* m_scene;
    opengl::Display* m_display;
    int m_width, m_height;

    DeviceContext* m_device_context;
    LaunchParam* m_launch_param;
};

};  // namespace optix