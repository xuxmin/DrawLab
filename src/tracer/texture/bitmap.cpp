#include "core/bitmap/bitmap.h"
#include "core/base/common.h"
#include "core/utils/timer.h"
#include "tracer/texture.h"
#include <spdlog/spdlog.h>
#include <string>

namespace drawlab {

class BitmapTexture : public Texture2D {
public:
    BitmapTexture(const PropertyList& props) : Texture2D(props) {
        m_filename = props.getString("filename");

        spdlog::info("Loading texture \"{}\" ...", m_filename);
        Timer timer;

        filesystem::path file = getFileResolver()->resolve(m_filename);
        if (!file.exists()) {
            spdlog::warn("Texture file \"{}\" could not be found!", m_filename);
        }
        m_bitmap = new Bitmap(m_filename);
        spdlog::info("Load donw. (took {})", timer.elapsedString());
    }

    Color3f eval(const Intersection& its) const {
        Point2f uv =
            Point2f(its.uv.x() * m_uvScale.x(), its.uv.y() * m_uvScale.y()) +
            m_uvOffset;
        return eval(uv);
    }

    Color3f eval(const Point2f& uv) const {
        throw Exception("BitmapTexture::eval(): not implemented!");
    }

    Color3f eval(const Point2f& uv, const Vector2f& d0,
                 const Vector2f& d1) const {
        throw Exception("BitmapTexture::eval(): not implemented!");
    }

    bool isConstant() const { return false; }

    Vector3i getResolution() const {
        return Vector3i(m_bitmap->getWidth(), m_bitmap->getHeight(), 0);
    }

    std::string toString() const {
        return tfm::format("BitmapTexture[filename=%s]", m_filename);
    }

    ~BitmapTexture() { delete m_bitmap; }

    std::shared_ptr<Bitmap> getBitmap(const Vector2i&) const {
        return std::shared_ptr<Bitmap>(m_bitmap);
    }

    const optix::Texture* getOptixTexture(optix::DeviceContext& context) const {
        std::string tex_id = m_filename + std::to_string(int(m_bitmap));
        const optix::Texture* texture = context.getTexture(tex_id);
        if (texture != nullptr) {
            return texture;
        }

        // Create a texture
        int width = m_bitmap->getWidth();
        int height = m_bitmap->getHeight();
        std::vector<unsigned char> temp(width * height * 4);
        for (int i = 0; i < width * height; i++) {
            temp[4 * i] = m_bitmap->getPtr()[3 * i];
            temp[4 * i + 1] = m_bitmap->getPtr()[3 * i + 1];
            temp[4 * i + 2] = m_bitmap->getPtr()[3 * i + 2];
            temp[4 * i + 3] = m_bitmap->getPtr()[3 * i + 2];
        }

        optix::Texture* optix_texture = new optix::Texture(
            width, height, 0, optix::CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA8,
            optix::CUDATextureFilterMode::CUDA_TEXTURE_LINEAR,
            optix::CUDATextureAddressMode::CUDA_TEXTURE_WRAP,
            optix::CUDATextureColorSpace::CUDA_COLOR_SPACE_LINEAR, temp.data());

        context.addTexture(tex_id, optix_texture);

        return optix_texture;
    }

protected:
    std::string m_filename;
    Bitmap* m_bitmap;
};

REGISTER_CLASS(BitmapTexture, "bitmap");

}  // namespace drawlab