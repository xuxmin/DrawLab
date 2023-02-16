#pragma once
#include <cuda_runtime.h>

namespace optix {

struct Microfacet {
    float4 kd;
    float ks;
    float alpha;
    float intIOR;
    float extIOR;
};

}  // namespace optix