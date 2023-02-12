#include "opengl/display.h"
#include "opengl/exception.h"

namespace opengl {

namespace {

    GLuint createGLShader(const std::string& source, GLuint shader_type) {
        GLuint shader = glCreateShader(shader_type);
        {
            const GLchar* source_data =
                reinterpret_cast<const GLchar*>(source.data());
            glShaderSource(shader, 1, &source_data, nullptr);
            glCompileShader(shader);

            GLint is_compiled = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
            if (is_compiled == GL_FALSE) {
                GLint max_length = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

                std::string info_log(max_length, '\0');
                GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
                glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

                glDeleteShader(shader);
                std::cerr << "Compilation of shader failed: " << info_log
                          << std::endl;
                return 0;
            }
        }
        GL_CHECK_ERRORS();
        return shader;
    }

    GLuint createGLProgram(const std::string& vert_source,
                           const std::string& frag_source) {
        GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
        if (vert_shader == 0)
            return 0;

        GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
        if (frag_shader == 0) {
            glDeleteShader(vert_shader);
            return 0;
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);

        GLint is_linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
        if (is_linked == GL_FALSE) {
            GLint max_length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log(max_length, '\0');
            GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
            glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
            std::cerr << "Linking of program failed: " << info_log << std::endl;

            glDeleteProgram(program);
            glDeleteShader(vert_shader);
            glDeleteShader(frag_shader);
            return 0;
        }

        glDetachShader(program, vert_shader);
        glDetachShader(program, frag_shader);
        GL_CHECK_ERRORS();

        return program;
    }

    GLint getGLUniformLocation(GLuint program, const std::string& name) {
        GLint loc = glGetUniformLocation(program, name.c_str());
        SUTIL_ASSERT_MSG(loc != -1,
                         "Failed to get uniform loc for '" + name + "'");
        return loc;
    }

}  // namespace

//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

const std::string Display::s_vertSource = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string Display::s_fragSource = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;
float toSRGB(float value) {
    if (value < 0.0031308)
        return 12.92 * value;
    return 1.055 * pow(value, 0.41666) - 0.055;
}
void main()
{
    color = texture( render_tex, UV ).xyz;

    color = vec3(toSRGB(color.x), toSRGB(color.y), toSRGB(color.z));
}
)";

Display::Display(BufferImageFormat format) : m_imageFormat(format) {
    GLuint m_vertex_array;
    GL_CHECK(glGenVertexArrays(1, &m_vertex_array));
    GL_CHECK(glBindVertexArray(m_vertex_array));

    m_program = createGLProgram(s_vertSource, s_fragSource);
    m_renderTexUniformLoc = getGLUniformLocation(m_program, "render_tex");

    GL_CHECK(glGenTextures(1, &m_renderTex));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_renderTex));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
    };

    GL_CHECK(glGenBuffers(1, &m_quadVertexBuffer));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quadVertexBuffer));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data),
                          g_quad_vertex_buffer_data, GL_STATIC_DRAW));

    GL_CHECK_ERRORS();
}

Display::~Display() {
    if (m_pbo != 0u) {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CHECK(glDeleteBuffers(1, &m_pbo));
    }
}

void Display::display(const int screen_res_x, const int screen_res_y,
                      const int framebuf_res_x, const int framebuf_res_y,
                      const unsigned int pbo) const {
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glViewport(0, 0, framebuf_res_x, framebuf_res_y));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(m_program));

    // Bind our texture in Texture Unit 0
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_renderTex));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));  // TODO!!!!!!

    size_t elmt_size = pixelFormatSize(m_imageFormat);
    if (elmt_size % 8 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if (m_imageFormat == BufferImageFormat::UNSIGNED_BYTE4) {
        // input is assumed to be in srgb since it is only 1 byte per channel in
        // size
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        convertToSrgb = false;
    } else if (m_imageFormat == BufferImageFormat::FLOAT3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res_x, screen_res_y, 0,
                     GL_RGB, GL_FLOAT, nullptr);

    else if (m_imageFormat == BufferImageFormat::FLOAT4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y,
                     0, GL_RGBA, GL_FLOAT, nullptr);

    else
        throw drawlab::Exception("Unknown buffer format");

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(m_renderTexUniformLoc, 0));

    // 1st attribute buffer : vertices
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quadVertexBuffer));
    GL_CHECK(
        glVertexAttribPointer(0,  // attribute 0. No particular reason for 0,
                                  // but must match the layout in the shader.
                              3,         // size
                              GL_FLOAT,  // type
                              GL_FALSE,  // normalized?
                              0,         // stride
                              (void*)0   // array buffer offset
                              ));

    GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    // Draw the triangles !
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0,
                          6));  // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK(glDisableVertexAttribArray(0));

    GL_CHECK_ERRORS();
}

size_t Display::pixelFormatSize(BufferImageFormat format) const {
    switch (format) {
        case BufferImageFormat::UNSIGNED_BYTE4: return sizeof(char) * 4;
        case BufferImageFormat::FLOAT3: return sizeof(float) * 3;
        case BufferImageFormat::FLOAT4: return sizeof(float) * 4;
        default:
            throw drawlab::Exception(
                "display::pixelFormatSize: Unrecognized buffer format");
    }
}

GLuint Display::getPBO(int width, int height, float* data) {
    if (m_pbo == 0u)
        glGenBuffers(1, &m_pbo);

    const size_t buffer_size = width * height * sizeof(float) * 3;

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, static_cast<void*>(data),
                 GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return m_pbo;
}

}  // namespace opengl
