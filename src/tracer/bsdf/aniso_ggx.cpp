#include "optix/material/aniso_ggx.h"
#include "core/base/common.h"
#include "core/math/frame.h"
#include "core/math/wrap.h"
#include "tracer/bsdf.h"
#include "tracer/texture.h"

namespace drawlab {

class AnisoGGX : public BSDF {
public:
    AnisoGGX(const PropertyList& propList) {
        Color3f pd = propList.getColor("pd", Color3f(0.5f));
        m_pd = new ConstantTexture(pd);

        Color3f ps = propList.getColor("ps", Color3f(0.5f));
        m_ps = new ConstantTexture(ps);

        Color3f axay = propList.getColor("axay", Color3f(0.5f));
        m_axay = new ConstantTexture(axay);

        m_normal = new ConstantTexture(Vector3f(0.5f, 0.5f, 1.f));
        m_tangent = new ConstantTexture(Vector3f(1.f, 0.5f, 0.5f));

        m_is_tangent_space = propList.getBoolean("is_tangent_space", true);
    }

    /// Evaluate the BRDF model
    Color3f eval(const BSDFQueryRecord& bRec) const {
        return Color3f(0.f);
    }

    /// Compute the density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord& bRec) const {
        return 0.f;
    }

    /// Draw a a sample from the BRDF model
    Color3f sample(BSDFQueryRecord& bRec, const Point2f& sample) const {
        return Color3f(0.f);
    }

    void addChild(Object* child) {
        if (child->getClassType() == ETexture) {
            if (child->getObjectName() == "pd")
                m_pd = static_cast<Texture*>(child);
            else if (child->getObjectName() == "ps")
                m_ps = static_cast<Texture*>(child);
            else if (child->getObjectName() == "axay")
                m_axay = static_cast<Texture*>(child);
            else if (child->getObjectName() == "normal")
                m_normal = static_cast<Texture*>(child);
            else if (child->getObjectName() == "tangent")
                m_tangent = static_cast<Texture*>(child);
        }
        else {
            spdlog::warn("The child({}) of aniso_ggx is not used!", child->getObjectName());
        }
    }

    bool isDiffuse() const { return true; }

    /// Return a human-readable summary
    std::string toString() const {
        return tfm::format("AnisoGGX[\n"
                           "  pd = %f,\n"
                           "  ps = %f,\n"
                           "  axay = %f,\n"
                           "  normal = %f,\n"
                           "  tangent = %f,\n"
                           "]",
                           m_pd->toString(), m_ps->toString(),
                           m_axay->toString(), m_normal->toString(),
                           m_tangent->toString());
    }

    EClassType getClassType() const { return EBSDF; }

    void createOptixMaterial(
        optix::Material& mat,
        std::vector<const optix::CUDATexture*>& textures) const {
        mat.type = optix::Material::ANISOGGX;
        mat.is_diffuse = true;
        mat.is_tangent_space = m_is_tangent_space;

        if (m_pd->isConstant()) {
            Color3f pd = m_pd->eval(Intersection());
            mat.aniso_ggx.pd = make_float3(pd[0], pd[1], pd[2]);
            mat.aniso_ggx.pd_tex = 0;
        }
        else {
            const optix::CUDATexture* tex = m_pd->createCUDATexture();
            textures.push_back(tex);
            mat.aniso_ggx.pd_tex = tex->getObject();
        }

        if (m_ps->isConstant()) {
            Color3f ps = m_ps->eval(Intersection());
            mat.aniso_ggx.ps = make_float3(ps[0], ps[1], ps[2]);
            mat.aniso_ggx.ps_tex = 0;
        }
        else {
            const optix::CUDATexture* tex = m_ps->createCUDATexture();
            textures.push_back(tex);
            mat.aniso_ggx.ps_tex = tex->getObject();
        }

        if (m_axay->isConstant()) {
            Color3f axay = m_axay->eval(Intersection());
            mat.aniso_ggx.axay = make_float2(axay[0], axay[1]);
            mat.aniso_ggx.axay_tex = 0;
        }
        else {
            const optix::CUDATexture* tex = m_axay->createCUDATexture();
            textures.push_back(tex);
            mat.aniso_ggx.axay_tex = tex->getObject();
        }

        if (m_normal->isConstant()) {
            Color3f n = m_normal->eval(Intersection());
            mat.normal_tex = 0;
        }
        else {
            const optix::CUDATexture* tex = m_normal->createCUDATexture();
            textures.push_back(tex);
            mat.normal_tex = tex->getObject();
        }

        if (m_tangent->isConstant()) {
            Color3f t = m_tangent->eval(Intersection());
            mat.tangent_tex = 0;
        }
        else {
            const optix::CUDATexture* tex = m_tangent->createCUDATexture();
            textures.push_back(tex);
            mat.tangent_tex = tex->getObject();
        }
    }

    optix::Material::Type getOptixMaterialType() const {
        return optix::Material::ANISOGGX;
    }

private:
    const Texture* m_pd;
    const Texture* m_ps;
    const Texture* m_axay;
    const Texture* m_normal;
    const Texture* m_tangent;

    bool m_is_tangent_space;
};

REGISTER_CLASS(AnisoGGX, "aniso_ggx");

}  // namespace drawlab
