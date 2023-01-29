#pragma once
#include "core/base/common.h"

namespace drawlab {

class CPURenderer {
public:
    static void render(Scene* scene, const std::string& filename,
                       const bool gui = false, const int thread_count = 4);

private:
    static void renderBlock(const Scene* scene, Sampler* sampler,
                            ImageBlock& block);
};

}  // namespace drawlab