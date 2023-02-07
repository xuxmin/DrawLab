#include "editor/utils.h"
#include "imgui/imgui.h"

namespace drawlab {

void displayText(const char* text, float x, float y) {
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::Begin("TextOverlayFG", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoInputs);
    ImGui::TextColored(ImColor(0.7f, 0.7f, 0.7f, 1.0f), "%s", text);
    ImGui::End();
}

void displayStats(std::chrono::duration<double>& state_update_time,
                  std::chrono::duration<double>& render_time,
                  std::chrono::duration<double>& display_time) {
    constexpr std::chrono::duration<double> display_update_min_interval_time(
        0.5);
    static int32_t total_subframe_count = 0;
    static int32_t last_update_frames = 0;
    static auto last_update_time = std::chrono::steady_clock::now();
    static char display_text[128];

    const auto cur_time = std::chrono::steady_clock::now();

    last_update_frames++;

    typedef std::chrono::duration<double, std::milli> durationMs;

    if (cur_time - last_update_time > display_update_min_interval_time ||
        total_subframe_count == 0) {
        sprintf(display_text,
                "%5.1f fps\n\n"
                "state update: %8.1f ms\n"
                "render      : %8.1f ms\n"
                "display     : %8.1f ms\n",
                last_update_frames /
                    std::chrono::duration<double>(cur_time - last_update_time)
                        .count(),
                (durationMs(state_update_time) / last_update_frames).count(),
                (durationMs(render_time) / last_update_frames).count(),
                (durationMs(display_time) / last_update_frames).count());

        last_update_time = cur_time;
        last_update_frames = 0;
        state_update_time = render_time = display_time =
            std::chrono::duration<double>::zero();
    }
    displayText(display_text, 10.0f, 10.0f);

    ++total_subframe_count;
}

}  // namespace drawlab