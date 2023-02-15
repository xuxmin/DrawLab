#include "optix/host/cuda_texture.h"
#include "optix/host/sutil.h"

namespace optix {

CUDATexture::CUDATexture(uint32_t width, uint32_t height, uint32_t linePitchInBytes,
                 CUDATexelFormat texelFormat, CUDATextureFilterMode filterMode,
                 CUDATextureAddressMode addressMode,
                 CUDATextureColorSpace colorSpace, const void* texels)
    : m_width(width), m_height(height), m_texture_object(0),
      m_line_pitch_in_bytes(linePitchInBytes) {
    uint32_t pitch = linePitchInBytes;
    if (pitch == 0) {
        switch (texelFormat) {
            case CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA8:
                pitch = width * sizeof(uchar4);
                break;
            case CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA32F:
                pitch = width * sizeof(float4);
                break;
            case CUDATexelFormat::CUDA_TEXEL_FORMAT_R8:
                pitch = width * sizeof(uint8_t);
                break;
            case CUDATexelFormat::CUDA_TEXEL_FORMAT_R32F:
                pitch = width * sizeof(float);
                break;
            default: throw Exception("Unknown texel format");
        }
    }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc;
    switch (texelFormat) {
        case CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA8:
            channel_desc = cudaCreateChannelDesc<uchar4>();
            break;
        case CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA32F:
            channel_desc = cudaCreateChannelDesc<float4>();
            break;
        case CUDATexelFormat::CUDA_TEXEL_FORMAT_R8:
            channel_desc = cudaCreateChannelDesc<uint8_t>();
            break;
        case CUDATexelFormat::CUDA_TEXEL_FORMAT_R32F:
            channel_desc = cudaCreateChannelDesc<float>();
            break;
        default: throw Exception("Unknown texel format");
    }

    CUDA_CHECK(cudaMallocArray(&m_texture_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(m_texture_array,
                                   /* offset */ 0, 0, texels, pitch, pitch,
                                   height, cudaMemcpyHostToDevice));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = m_texture_array;

    cudaTextureDesc tex_desc = {};
    if (addressMode == CUDATextureAddressMode::CUDA_TEXTURE_BORDER) {
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
    }
    else if (addressMode == CUDATextureAddressMode::CUDA_TEXTURE_CLAMP) {
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
    }
    else if (addressMode == CUDATextureAddressMode::CUDA_TEXTURE_WRAP) {
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
    }
    else {
        tex_desc.addressMode[0] = cudaAddressModeMirror;
        tex_desc.addressMode[1] = cudaAddressModeMirror;
    }
    tex_desc.filterMode =
        filterMode == CUDATextureFilterMode::CUDA_TEXTURE_NEAREST ?
            cudaFilterModePoint :
            cudaFilterModeLinear;
    tex_desc.readMode =
        ((texelFormat == CUDATexelFormat::CUDA_TEXEL_FORMAT_R8) ||
         (texelFormat == CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA8)) ?
            cudaReadModeNormalizedFloat :
            cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB =
        (colorSpace == CUDATextureColorSpace::CUDA_COLOR_SPACE_SRGB);

    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&m_texture_object, &res_desc, &tex_desc,
                                       nullptr));
}

const cudaTextureObject_t CUDATexture::getObject() const {
    return m_texture_object;
}

CUDATexture::~CUDATexture() {
    cudaDestroyTextureObject(m_texture_object);
    cudaFreeArray(m_texture_array);
}

}  // namespace optix