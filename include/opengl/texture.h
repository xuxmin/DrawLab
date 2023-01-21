#pragma once

#include "glad/glad.h"
#include <vector>

namespace opengl {

class Texture {
public:
    Texture();
    Texture(GLuint wrap_s, GLuint wrap_t, GLuint filter_min, GLuint filter_max,
            GLuint internal_format, GLuint image_format);
    ~Texture();

    void createImageTexture(GLuint width, GLuint height,
                            const std::vector<float>& data);
    void bind() const;
    GLuint getId() const;

private:
    GLuint m_width, m_height;  // Width and height of loaded image in pixels

    GLuint m_wrapS;      // Wrapping mode on S axis
    GLuint m_wrapT;      // Wrapping mode on T axis
    GLuint m_filterMin;  // Filtering mode if texture pixels < screen pixels
    GLuint m_filterMax;  // Filtering mode if texture pixels > screen pixels

    GLuint m_textureId;  // The ID of the texture object

    GLuint m_internalFormat;  // Format of texture object
    GLuint m_imageFormat;     // Format of loaded image

    void createMipmap() const;
};

}