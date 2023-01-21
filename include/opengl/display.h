#pragma once

#include "glad/glad.h"
#include <string>


namespace opengl {

class Display {
public:
    enum BufferImageFormat { UNSIGNED_BYTE4, FLOAT4, FLOAT3 };

    Display(BufferImageFormat format = BufferImageFormat::FLOAT3);

    void display(const int screenResX, const int screenResY,
                 const int framebufResX, const int framebuf_ResY,
                 const unsigned int pbo) const;

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