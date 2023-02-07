#include "glad/glad.h"
#include "editor/gui.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <spdlog/spdlog.h>

namespace drawlab {

GLFWwindow* GUI::window;
int GUI::window_width;
int GUI::window_height;

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
    float s = std::max(mode->width / (float)size[0], mode->height / (float)size[1]);
    m_scale = s > 2 ? s / 2 : 1;
    window_width = (int)(window_width * m_scale);
    window_height = (int)(window_height * m_scale);
}

void GUI::init() {
    // initialize glfw
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
    window = glfwCreateWindow(window_width, window_height, "drawlab", nullptr, nullptr);
    if (!window) {
        spdlog::critical("Error: could not create window!");
        glfwTerminate();
        exit(1);
    }

    // set context
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize gl
    if (!gladLoadGL()) {
        spdlog::critical("Failed to initialize GL");
        glfwTerminate();
        exit(1);
    }

    // initialize renderer if already set
    if (m_renderer) {
        m_renderer->init();
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
    ImGui::GetStyle().ScaleAllSizes(m_scale);
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
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Controller");

    if (m_renderer) {
        m_renderer->render();
    }

    ImGui::End();
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

}  // namespace drawlab