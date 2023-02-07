#pragma once
#include "core/base/common.h"
#include "editor/renderer.h"
#include "opengl/display.h"
#include <thread>

namespace drawlab {

class CPURenderer : public Renderer {
public:
    CPURenderer(Scene* scene);

    void init();
    void render();
    void resize(size_t w, size_t h);
    void keyEvent(char key);
    void cursorEvent(float x, float y, unsigned char keys);
    void scrollEvent(float offset_x, float offset_y);
    void mouseButtonEvent(int button, int event);

public:
    void renderAsync(const std::string& filename, bool gui = false,
                     const int thread_count = 4);

private:
    static void renderBlock(const Scene* scene, Sampler* sampler,
                            ImageBlock& block);

private:
    Scene* m_scene;
    ImageBlock* m_block;
    opengl::Display* m_display;
};

}  // namespace drawlab