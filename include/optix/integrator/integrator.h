#pragma once

#include <cuda_runtime.h>

namespace optix {

struct Integrator {
    enum Type { 
        PATH = 0,
        NORMAL = 1,
        INTEGRATOR_NUM = 2
    };
    Type type;

    Integrator() {
        type = PATH;
    }
};

}  // namespace optix