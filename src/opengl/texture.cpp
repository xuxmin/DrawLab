#include "opengl/texture.h"

namespace opengl {

Texture::Texture()
    : m_width(0), m_height(0), m_internalFormat(GL_RGBA),
      m_imageFormat(GL_RGBA), m_wrapS(GL_REPEAT), m_wrapT(GL_REPEAT),
      m_filterMin(GL_LINEAR), m_filterMax(GL_LINEAR) {
    glGenTextures(1, &this->m_textureId);
}

Texture::Texture(GLuint wrap_s, GLuint wrap_t, GLuint filter_min,
                 GLuint filter_max, GLuint internal_format, GLuint image_format)
    : m_width(0), m_height(0), m_internalFormat(internal_format),
      m_imageFormat(image_format), m_wrapS(wrap_s), m_wrapT(wrap_t),
      m_filterMin(filter_min), m_filterMax(filter_max) {
    glGenTextures(1, &this->m_textureId);
}

void Texture::createImageTexture(GLuint w, GLuint h,
                                 const std::vector<float>& data) {
    this->m_width = w;
    this->m_height = h;

    glBindTexture(GL_TEXTURE_2D, this->m_textureId);
    glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, m_width, m_height, 0,
                 m_imageFormat, GL_FLOAT, &data[0]);
    glBindTexture(GL_TEXTURE_2D, 0);

    this->createMipmap();
}

void Texture::createMipmap() const {
    glBindTexture(GL_TEXTURE_2D, this->m_textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, this->m_wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, this->m_wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, this->m_filterMin);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, this->m_filterMax);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::bind() const { glBindTexture(GL_TEXTURE_2D, this->m_textureId); }

GLuint Texture::getId() const { return this->m_textureId; }

}  // namespace opengl
