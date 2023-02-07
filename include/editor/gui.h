#pragma once

#include "core/base/common.h"
#include "editor/renderer.h"
#include <GLFW/glfw3.h>

namespace drawlab {

class GUI {
public:
    GUI(int width = 1920, int height = 1080);
    ~GUI();

    /// @brief Initialize UI and bind the event.
    void init();

    /// @brief Start rendering
    void start();

    void setRenderer(Renderer* renderer) { m_renderer = renderer; }

    static int window_width;
    static int window_height;

private:
    void update();

    // static void errCallback(int error, const char* description);
    // static void keyCallback(GLFWwindow* window, int key, int scancode,
    //                         int action, int mods);
    // static void resizeCallback(GLFWwindow* window, int width, int height);
    // static void cursorCallback(GLFWwindow* window, double xpos, double ypos);
    // static void scrollCallback(GLFWwindow* window, double xoffset,
    //                            double yoffset);
    // static void mouseButtonCallback(GLFWwindow* window, int button, int
    // action,
    //                                 int mods);

    void scaleWindow();

    // imgui
    static void beginFrameImGui();
    static void endFrameImGui();

private:
    float m_scale;

    Renderer* m_renderer;

    static GLFWwindow* window;
};

}  // namespace drawlab