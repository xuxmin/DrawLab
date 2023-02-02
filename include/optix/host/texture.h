#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>


namespace optix {

enum class CUDATexelFormat {
    CUDA_TEXEL_FORMAT_RGBA8 = 0,
    CUDA_TEXEL_FORMAT_RGBA32F,
    CUDA_TEXEL_FORMAT_R8,
    CUDA_TEXEL_FORMAT_R32F
};

enum class CUDATextureFilterMode {
    CUDA_TEXTURE_NEAREST = 0,
    CUDA_TEXTURE_LINEAR
};

enum class CUDATextureAddressMode {
    CUDA_TEXTURE_WRAP = 0,
    CUDA_TEXTURE_CLAMP,
    CUDA_TEXTURE_BORDER,
    CUDA_TEXTURE_MIRROR
};

enum class CUDATextureColorSpace {
    CUDA_COLOR_SPACE_LINEAR = 0,
    CUDA_COLOR_SPACE_SRGB
};

class Texture {
public:
    Texture(uint32_t width, uint32_t height,
            uint32_t linePitchInBytes,
            CUDATexelFormat texelFormat,
            CUDATextureFilterMode filterMode,
            CUDATextureAddressMode addressMode,
            CUDATextureColorSpace colorSpace, const void* texels);
    
    ~Texture();

    const cudaTextureObject_t getObject() const;

private:
    cudaTextureObject_t m_texture_object;
    cudaArray_t m_texture_array;

    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_line_pitch_in_bytes;
};

}  // namespace optix