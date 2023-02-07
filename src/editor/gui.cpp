#include "glad/glad.h"
#include "editor/gui.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <spdlog/spdlog.h>
#include <tinyformat/tinyformat.h>

namespace drawlab {

GLFWwindow* GUI::window;
int GUI::window_width;
int GUI::window_height;
Renderer* GUI::renderer = nullptr;
float GUI::window_scale = 1.f;

GUI::GUI(int width, int height) {
    window_width = width;
    window_height = height;
}

GUI::~GUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void GUI::scaleWindow() {
    Vector2i size = Vector2i(window_width, window_height);
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    float s =
        std::max(mode->width / (float)size[0], mode->height / (float)size[1]);
    window_scale = s > 2 ? s / 2 : 1;
    window_width = (int)(window_width * window_scale);
    window_height = (int)(window_height * window_scale);
}

void GUI::init() {
    // initialize glfw
    glfwSetErrorCallback(GUI::errCallback);
    if (!glfwInit()) {
        spdlog::critical("Error: could not initialize GLFW!");
        exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create window
    scaleWindow();
    window = glfwCreateWindow(window_width, window_height, "drawlab", nullptr,
                              nullptr);
    if (!window) {
        spdlog::critical("Error: could not create window!");
        glfwTerminate();
        exit(1);
    }
    spdlog::info("[GUI] Create window with size: {} x {}", window_width, window_height);

    // set context
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // framebuffer event callbacks
    glfwSetFramebufferSizeCallback(window, GUI::resizeCallback);

    // key event callbacks
    glfwSetKeyCallback(window, GUI::keyCallback);

    // cursor event callbacks
    glfwSetCursorPosCallback(window, GUI::cursorCallback);

    // wheel event callbacks
    glfwSetScrollCallback(window, GUI::scrollCallback);

    // mouse button callbacks
    glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
    glfwSetMouseButtonCallback(window, GUI::mouseButtonCallback);

    // initialize gl
    if (!gladLoadGL()) {
        spdlog::critical("Failed to initialize GL");
        glfwTerminate();
        exit(1);
    }

    // initialize renderer if already set
    if (renderer) {
        renderer->init();
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init();

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can
    // also load multiple fonts and use ImGui::PushFont()/PopFont() to select
    // them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you
    // need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please
    // handle those errors in your application (e.g. use an assertion, or
    // display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and
    // stored into a texture when calling
    // ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame
    // below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string
    // literal you need to write a double backslash \\ !
    // io.Fonts->AddFontDefault();
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontDefault();

    ImGui::GetStyle().WindowBorderSize = 0.0f;
    ImGui::GetStyle().ScaleAllSizes(window_scale);
}

void GUI::start() {
    while (!glfwWindowShouldClose(window)) {
        update();
    }
}

void GUI::update() {
    glfwPollEvents();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    beginFrameImGui();

    if (renderer) {
        renderer->render();
    }

    endFrameImGui();
    glfwSwapBuffers(window);
}

void GUI::beginFrameImGui() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::endFrameImGui() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::errCallback(int error, const char* description) {
    spdlog::critical("GLFW Error " + std::to_string(error) + ": " +
                     description);
}

void GUI::keyCallback(GLFWwindow* window, int key, int scancode, int action,
                      int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        }
        else {
            if (renderer)
                renderer->keyEvent(key);
        }
    }
}

void GUI::resizeCallback(GLFWwindow* window, int width, int height) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);

    if (renderer)
        renderer->resize(int(w / window_scale), int(h / window_scale));
}

void GUI::cursorCallback(GLFWwindow* window, double xpos, double ypos) {
    float cursor_x = (float)xpos;
    float cursor_y = (float)ypos;
    renderer->cursorEvent(cursor_x, cursor_y);
}

void GUI::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    renderer->scrollEvent((float)xoffset, (float)yoffset);
}

void GUI::mouseButtonCallback(GLFWwindow* window, int button, int action,
                              int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    renderer->mouseButtonEvent(button, action, (float)xpos, (float)ypos);
}

}  // namespace drawlab