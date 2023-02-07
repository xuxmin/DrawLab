#pragma once

#include <chrono>

namespace drawlab {

void displayText(const char* text, float x, float y);

void displayStats(std::chrono::duration<double>& state_update_time,
                  std::chrono::duration<double>& render_time,
                  std::chrono::duration<double>& display_time);

}  // namespace optix