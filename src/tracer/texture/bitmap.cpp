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
        m_bitmap = std::make_shared<Bitmap>(m_filename);
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
        return tfm::format("BitmapTexture[\n"
                           "    filename=%s\n"
                           "    size=[%d, %d]\n"
                           "]",
                           m_filename, m_bitmap->getWidth(),
                           m_bitmap->getHeight());
    }

    ~BitmapTexture() {}

    std::shared_ptr<Bitmap> getBitmap(const Vector2i&) const {
        return m_bitmap;
    }

    const optix::CUDATexture* createCUDATexture() const {

        // Create a texture
        int width = m_bitmap->getWidth();
        int height = m_bitmap->getHeight();

        optix::CUDATexture* texture;
        if (m_bitmap->getPixelFormat() == Bitmap::PixelFormat::UCHAR3) {
            std::vector<unsigned char> temp(width * height * 4);
            for (int i = 0; i < width * height; i++) {
                temp[4 * i] = m_bitmap->getPtr()[3 * i];
                temp[4 * i + 1] = m_bitmap->getPtr()[3 * i + 1];
                temp[4 * i + 2] = m_bitmap->getPtr()[3 * i + 2];
                temp[4 * i + 3] = m_bitmap->getPtr()[3 * i + 2];
            }

            texture = new optix::CUDATexture(
                width, height, 0, optix::CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA8,
                optix::CUDATextureFilterMode::CUDA_TEXTURE_LINEAR,
                optix::CUDATextureAddressMode::CUDA_TEXTURE_WRAP,
                optix::CUDATextureColorSpace::CUDA_COLOR_SPACE_LINEAR, temp.data());
        }
        else {
            m_bitmap->flipud();
            std::vector<float> temp(width * height * 4);
            for (int i = 0; i < width * height; i++) {
                temp[4 * i] = m_bitmap->getPtr()[3 * i];
                temp[4 * i + 1] = m_bitmap->getPtr()[3 * i + 1];
                temp[4 * i + 2] = m_bitmap->getPtr()[3 * i + 2];
                temp[4 * i + 3] = m_bitmap->getPtr()[3 * i + 2];
            }

            texture = new optix::CUDATexture(
                width, height, 0, optix::CUDATexelFormat::CUDA_TEXEL_FORMAT_RGBA32F,
                optix::CUDATextureFilterMode::CUDA_TEXTURE_LINEAR,
                optix::CUDATextureAddressMode::CUDA_TEXTURE_WRAP,
                optix::CUDATextureColorSpace::CUDA_COLOR_SPACE_LINEAR, temp.data());
        }

        return texture;
    }

protected:
    std::string m_filename;
    std::shared_ptr<Bitmap> m_bitmap;
};

REGISTER_CLASS(BitmapTexture, "bitmap");

}  // namespace drawlab