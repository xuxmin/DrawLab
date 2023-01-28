#include "core/bitmap/bitmap.h"
#include "core/base/common.h"
#include "core/utils/timer.h"
#include "tracer/texture.h"
#include <string>

namespace drawlab {

class BitmapTexture : public Texture2D {
public:
    BitmapTexture(const PropertyList& props) : Texture2D(props) {
        m_filename = props.getString("filename");

        cout << "Loading texture \"" << m_filename << "\" .. ";
        Timer timer;

        filesystem::path file = getFileResolver()->resolve(m_filename);
        if (!file.exists()) {
            cout << "Texture file \"" << m_filename << "\" could not be found!"
                 << endl;
        }
        m_bitmap = new Bitmap(m_filename);
        cout << "done."
             << "(took " << timer.elapsedString() << ")" << endl;
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

protected:
    std::string m_filename;
    Bitmap* m_bitmap;
};

REGISTER_CLASS(BitmapTexture, "bitmap");

}  // namespace drawlab