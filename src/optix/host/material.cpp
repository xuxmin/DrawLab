#include "optix/host/material.h"

namespace optix {


Material::Material(const DeviceContext& context, std::string cu_file) {

    // Create Module
    size_t ptxSize = 0;
    const char* ptxCode = optix::getInputData("device_programs.cu", ptxSize);
}

}