#pragma once

// optix 7
#include <cuda_runtime.h>
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include <string>

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

namespace optix {

class Exception : public std::runtime_error {
public:
    Exception(const char* msg) : std::runtime_error(msg) {}

    Exception(std::string msg) : std::runtime_error(msg.c_str()) {}

    Exception(OptixResult res, const char* msg)
        : std::runtime_error(createMessage(res, msg).c_str()) {}

private:
    std::string createMessage(OptixResult res, const char* msg) {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw optix::Exception(res, ss.str().c_str());                            \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                  \
    do {                                                                       \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof(log); /* reset sizeof_log for future calls */      \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n"                                      \
               << log                                                          \
               << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "")     \
               << "\n";                                                        \
            throw optix::Exception(res, ss.str().c_str());                            \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString(error) << "' (" __FILE__ << ":"           \
               << __LINE__ << ")\n";                                           \
            throw optix::Exception(ss.str().c_str());                                 \
        }                                                                      \
    } while (0)

#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString(error) << "' (" __FILE__ << ":"           \
               << __LINE__ << ")\n";                                           \
            throw optix::Exception(ss.str().c_str());                                 \
        }                                                                      \
    } while (0)

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW(call)                                               \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA call (" << #call << " ) failed with error: '"   \
                      << cudaGetErrorString(error) << "' (" __FILE__ << ":"    \
                      << __LINE__ << ")\n";                                    \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)

/// @brief Get input data, either pre-compiled with NVCC or JIT compiled by
/// NVRTC.
/// @param path cuda c file path relative to project root.
/// @param dataSize
/// @param log (Optional) pointer to compiler log string. If *log == NULL there
/// is no output. Only valid until the next getInputData call
/// @return
const char* getInputData(const char* filepath, size_t& dataSize,
                         const char** log = NULL);

void initOptix();
}  // namespace optix
