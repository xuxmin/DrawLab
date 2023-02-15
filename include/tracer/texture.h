#include "core/parser/object.h"
#include "core/parser/proplist.h"
#include "tracer/mesh.h"
#include <memory>
#include "optix/host/cuda_texture.h"
#include "optix/host/device_context.h"


namespace drawlab {

/**
 * \brief Base class of all textures. Computes values for an arbitrary surface
 * point. \ref Texture2D is a specialization to UV-based textures.
 */
class Texture : public Object {
public:

    /**
     * \brief Return the texture value at \c its
     */
    virtual Color3f eval(const Intersection &its) const;

    /// Return the resolution in pixels, if applicable
    virtual Vector3i getResolution() const;

    /// Return whether the texture takes on a constant value everywhere
    virtual bool isConstant() const;

    /**
     * \brief Return a bitmap representation of the texture
     *
     * When the class implementing this interface is a bitmap-backed texture,
     * this function directly returns the underlying bitmap. When it is procedural,
     * a bitmap version must first be generated. In this case, the parameter
     * \ref sizeHint is used to control the target size. The default
     * value <tt>-1, -1</tt> allows the implementation to choose a suitable
     * size by itself.
     */
    virtual std::shared_ptr<Bitmap> getBitmap(const Vector2i &sizeHint = Vector2i(-1, -1)) const;

    /**
     * \brief Return the type of object (i.e. Mesh/Camera/etc.)
     * provided by this instance
     * */
    EClassType getClassType() const { return ETexture; }

    virtual const optix::CUDATexture* createCUDATexture() const;

protected:
    Texture(const PropertyList &props);
    Texture();
    virtual ~Texture();
};

class Texture2D : public Texture {
public:
    /**
     * \brief Return the texture value at \c its
     */
    virtual Color3f eval(const Intersection &its) const;

    /// Unfiltered texture lookup -- Texture2D subclasses must provide this function
    virtual Color3f eval(const Point2f &uv) const = 0;

    /// Filtered texture lookup -- Texture2D subclasses must provide this function
    virtual Color3f eval(const Point2f &uv, const Vector2f &d0,
            const Vector2f &d1) const = 0;

    /**
     * \brief Return a bitmap representation of the texture
     *
     * When the class implementing this interface is a bitmap-backed texture,
     * this function directly returns the underlying bitmap. When it is procedural,
     * a bitmap version must first be generated. In this case, the parameter
     * \ref sizeHint is used to control the target size. The default
     * value <tt>-1, -1</tt> allows the implementation to choose a suitable
     * size by itself.
     */
    virtual std::shared_ptr<Bitmap> getBitmap(const Vector2i &sizeHint = Vector2i(-1, -1)) const;

protected:
    Texture2D(const PropertyList &props);
    virtual ~Texture2D();

protected:
    Point2f m_uvOffset;
    Vector2f m_uvScale;
};


class ConstantTexture : public Texture {

public:
    ConstantTexture(Color3f color):Texture() {
        m_constant = color;
    }

    Color3f eval(const Intersection &its) const {
        return Color3f(m_constant[0], m_constant[1], m_constant[2]);
    }

    bool isConstant() const {
        return true;
    }

    Vector3i getResolution() const {
        return Vector3i(1, 0, 0);
    }

    std::shared_ptr<Bitmap> getBitmap(const Vector2i &sizeHint) const {
        throw Exception("ConstantTexture::getBitmap(): not implemented!");
    }

    std::string toString() const {
        return tfm::format("ConstantTexture[constant=%s]", m_constant.toString());
    }

private:
    Color3f m_constant;
};

}