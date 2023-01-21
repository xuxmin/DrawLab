#include <glad/glad.h>
#include <sstream>
#include "core/base/exception.h"


namespace opengl {

#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK(call)                                                     \
        do {                                                                   \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if (err != GL_NO_ERROR) {                                          \
                std::stringstream ss;                                          \
                ss << "GL error " << getGLErrorString(err) << " at "           \
                   << __FILE__ << "(" << __LINE__ << "): " << #call            \
                   << std::endl;                                               \
                std::cerr << ss.str() << std::endl;                            \
                throw drawlab::Exception(ss.str());                            \
            }                                                                  \
        } while (0)

#    define GL_CHECK_ERRORS()                                                  \
        do {                                                                   \
            GLenum err = glGetError();                                         \
            if (err != GL_NO_ERROR) {                                          \
                std::stringstream ss;                                          \
                ss << "GL error " << getGLErrorString(err) << " at "           \
                   << __FILE__ << "(" << __LINE__ << ")";                      \
                std::cerr << ss.str() << std::endl;                            \
                throw drawlab::Exception(ss.str());                            \
            }                                                                  \
        } while (0)

#else
#    define GL_CHECK(call)                                                     \
        do {                                                                   \
            call;                                                              \
        } while (0)
#    define GL_CHECK_ERRORS()                                                  \
        do {                                                                   \
            ;                                                                  \
        } while (0)
#endif

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
            throw drawlab::Exception(ss.str());                                \
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
            throw drawlab::Exception(ss.str());                                \
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

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define SUTIL_ASSERT(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw drawlab::Exception(ss.str());                                \
        }                                                                      \
    } while (0)

#define SUTIL_ASSERT_MSG(cond, msg)                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << (msg) << ": " << __FILE__ << " (" << __LINE__                \
               << "): " << #cond;                                              \
            throw drawlab::Exception(ss.str());                                \
        }                                                                      \
    } while (0)

inline const char* getGLErrorString(GLenum error) {
    switch (error) {
        case GL_NO_ERROR: return "No error";
        case GL_INVALID_ENUM: return "Invalid enum";
        case GL_INVALID_VALUE: return "Invalid value";
        case GL_INVALID_OPERATION: return "Invalid operation";
        // case GL_STACK_OVERFLOW:      return "Stack overflow";
        // case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY: return "Out of memory";
        // case GL_TABLE_TOO_LARGE:     return "Table too large";
        default: return "Unknown GL error";
    }
}

inline void checkGLError() {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::ostringstream oss;
        do {
            oss << "GL error: " << getGLErrorString(err) << "\n";
            err = glGetError();
        } while (err != GL_NO_ERROR);

        throw drawlab::Exception(oss.str());
    }
}

}  // namespace opengl
