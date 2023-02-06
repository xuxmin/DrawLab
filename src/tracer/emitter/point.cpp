#include "tracer/emitter.h"

namespace drawlab {


/**
 * Point Light can be seen as a very small ball
 * 
*/
class PointLight final : public Emitter {
public:
    PointLight(const PropertyList& propList) {
        m_intensity = propList.getColor("intensity");
        m_position = propList.getPoint("position");
    }

    void sampleDirection(const Point2f& sample, const Intersection& its,
                         DirectionSample& ds,
                         Color3f& spectrum) const override {
        ds = DirectionSample(its.p, m_position, Normal3f(0.f));
        ds.pdf = 1;

        float inv_dist = 1.0 / (float)ds.dist;
        spectrum = m_intensity * inv_dist * inv_dist;
    }

    float pdfDirection(const DirectionSample& ds) const override {
        return 0;
    }

    Color3f eval(const Intersection& its, const Vector3f wi) const {
        return Color3f(0.f);
    }

    std::string toString() const {
        return tfm::format("PointLight[\n"
                           "m_intensity = \"%f,%f,%f\"\n"
                           "m_position = \"%f,%f,%f\"\n"
                           "]",
                           m_intensity.x(), m_intensity.y(), m_intensity.z(),
                           m_position.x(), m_position.y(), m_position.z());
    }

    void getOptixLight(optix::Light& light) const {
        light.type = optix::Light::Type::POINT;
        light.point.intensity = make_float3(m_intensity[0], m_intensity[1], m_intensity[2]);
        light.point.position = make_float3(m_position[0], m_position[1], m_position[2]);
    }

private:
    Color3f m_intensity;
    Point3f m_position;
};

REGISTER_CLASS(PointLight, "point");

}  // namespace drawlab