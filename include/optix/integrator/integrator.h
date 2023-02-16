#pragma once

#include <cuda_runtime.h>

namespace optix {

struct Integrator {
    enum Type { PATH = 0, INTEGRATOR_NUM };
    Type type;

    Integrator() {
        type = Type::PATH;
    }
};

}  // namespace optix