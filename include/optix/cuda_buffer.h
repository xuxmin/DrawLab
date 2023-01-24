#pragma once

#include "optix/sutil.h"
#include <assert.h>
#include <vector>

namespace optix {

/**
 * Simple wrapper for crating, and managing a device-side
 * CUDA buffer.
 */
class CUDABuffer {
public:
    CUDABuffer() : m_device_ptr(nullptr), m_size_in_bytes(0) {}

    inline CUdeviceptr d_pointer() const { return (CUdeviceptr)m_device_ptr; }

    /// @brief Resize buffer to a given number of bytes
    void resize(size_t s) {
        if (m_device_ptr)
            free();
        alloc(s);
    }

    /// @brief Allocate s bytes memory
    void alloc(size_t s) {
        assert(m_device_ptr == nullptr);
        this->m_size_in_bytes = s;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                              m_size_in_bytes));
    }

    /// @brief Free memory
    void free() {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
        m_device_ptr = nullptr;
        m_size_in_bytes = 0;
    }

    template <typename T> void allocAndUpload(const std::vector<T>& data) {
        alloc(data.size() * sizeof(T));
        upload((const T*)data.data(), data.size());
    }

    template <typename T> void upload(const T* t, size_t count) {
        assert(m_device_ptr != nullptr);
        assert(m_size_in_bytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(m_device_ptr, (void*)(t),
                              count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T> void download(T* t, size_t count) {
        assert(m_device_ptr != nullptr);
        assert(m_size_in_bytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void*)t, m_device_ptr, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

public:
    void* m_device_ptr = nullptr;
    size_t m_size_in_bytes = 0;
};

};  // namespace optix