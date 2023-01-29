#include "editor/gui.h"
#include "glad/glad.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <spdlog/spdlog.h>

namespace drawlab {

GLFWwindow* GUI::window;

GUI::~GUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    delete m_display;
}

void GUI::init() {
    // initialize glfw
    if (!glfwInit()) {
        spdlog::critical("Error: could not initialize GLFW!");
        exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                   GL_TRUE);  // To make Apple happy -- should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    Vector2i size = m_block.getSize();

    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    float s =
        std::max(mode->width / (float)size[0], mode->height / (float)size[1]);
    m_scale = s > 2 ? s / 2 : 1;

    m_width = (int)(size[0] * m_scale);
    m_height = (int)(size[1] * m_scale);

    // create window
    window = glfwCreateWindow(m_width, m_height, "drawlab", nullptr, nullptr);
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

    m_display = new opengl::Display(opengl::Display::BufferImageFormat::FLOAT3);
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

    m_block.lock();
    Bitmap* bitmap = m_block.toBitmap();
    bitmap->flipud();
    bitmap->resize(m_height, m_width);
    unsigned int pbo = m_display->getPBO(m_width, m_height, bitmap->getPtr());
    m_display->display(m_width, m_height, m_width, m_height, pbo);
    delete bitmap;
    m_block.unlock();

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