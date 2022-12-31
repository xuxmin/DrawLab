#pragma once

#include "core/base/common.h"
#include "core/bitmap/bitmap.h"
#include "editor/block.h"
#include "opengl/display.h"

#include <GLFW/glfw3.h>

namespace drawlab {

class GUI {
public:
    GUI(const ImageBlock& block) : m_block(block){};
    ~GUI();

    /// @brief Initialize UI and bind the event.
    void init();

    /// @brief Start rendering
    void start();

private:
    const ImageBlock& m_block;

    int m_width;
    int m_height;
    float m_scale;

    opengl::Display* m_display;

    static GLFWwindow* window;

    void update();

    // imgui
    static void beginFrameImGui();
    static void endFrameImGui();
};

}  // namespace drawlab