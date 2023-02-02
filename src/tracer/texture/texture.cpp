#include "tracer/texture.h"
#include "core/bitmap/bitmap.h"


namespace drawlab {

Texture::Texture(const PropertyList &props) {

}

Texture::Texture() {

}

Vector3i Texture::getResolution() const {
    return Vector3i(0);
}

Color3f Texture::eval(const Intersection &its) const { 
    throw Exception("Texture::eval(): not implemented!");
}

bool Texture::isConstant() const { 
    throw Exception("Texture::isConstant(): not implemented!");
}

std::shared_ptr<Bitmap> Texture::getBitmap(const Vector2i &) const {
    throw Exception("Texture::getBitmap(): not implemented!");
}

const optix::Texture* Texture::getOptixTexture(optix::DeviceContext& context) const {
    throw Exception("Texture::getOptixTexture(): not implemented!");
}

Texture::~Texture() { }

Texture2D::Texture2D(const PropertyList &props) : Texture(props) {
    if (props.getString("coordinates", "uv") == "uv") {
        m_uvOffset = Point2f(
            props.getFloat("uoffset", 0.0f),
            props.getFloat("voffset", 0.0f)
        );
        float uvscale = props.getFloat("uvscale", 1.0f);
        m_uvScale = Vector2f(
            props.getFloat("uscale", uvscale),
            props.getFloat("vscale", uvscale)
        );
    } else {
        throw Exception("Only UV coordinates are supported at the moment!");
    }
}

Texture2D::~Texture2D() {
}

Color3f Texture2D::eval(const Intersection &its) const {
    Point2f uv = Point2f(its.uv.x() * m_uvScale.x(), its.uv.y() * m_uvScale.y()) + m_uvOffset;
    return eval(uv);
}

std::shared_ptr<Bitmap> Texture2D::getBitmap(const Vector2i &sizeHint) const {
    Vector2i res(sizeHint);
    if (res.x() <= 0 || res.y() <= 0)
        res = Vector2i(32);

    float invX = 1.0f / res.x(), invY = 1.0f / res.y();

    std::shared_ptr<Bitmap> bitmap = std::make_shared<Bitmap>(res.x(), res.y());

    for (int y = 0; y < res.y(); y++) {
        for (int x = 0; x < res.x(); x++) {
            Color3f color = eval(Point2f((x + 0.5f) * invX, (y + 0.5f) * invY));
            bitmap->setPixel(y, x, color);
        }
    }
    return bitmap;
}

}