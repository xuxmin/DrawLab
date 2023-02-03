#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace optix {

static __forceinline__ __device__ void* unpackPointer(unsigned int i0,
                                                      unsigned int i1) {
    const unsigned long long uptr =
        static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0,
                                                   unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T> static __forceinline__ __device__ T* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

}  // namespace optix