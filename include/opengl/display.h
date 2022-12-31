#pragma once

#include "core/base/common.h"
#include "glad/glad.h"

namespace opengl {

class Display {
public:
    enum BufferImageFormat { UNSIGNED_BYTE4, FLOAT4, FLOAT3 };

    Display(BufferImageFormat format = BufferImageFormat::FLOAT3);

    void display(const int32_t screenResX, const int32_t screenResY,
                 const int32_t framebufResX, const int32_t framebuf_ResY,
                 const uint32_t pbo) const;

    GLuint getPBO(int width, int height, float* data);

private:
    GLuint m_pbo = 0u;
    GLuint m_renderTex = 0u;
    GLuint m_program = 0u;
    GLint m_renderTexUniformLoc = -1;
    GLuint m_quadVertexBuffer = 0;
    BufferImageFormat m_imageFormat;

    static const std::string s_vertSource;
    static const std::string s_fragSource;

    size_t pixelFormatSize(BufferImageFormat format) const;
};

}  // namespace opengl