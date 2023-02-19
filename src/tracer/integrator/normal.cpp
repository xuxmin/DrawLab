#include "tracer/integrator.h"
#include "tracer/scene.h"

namespace drawlab {

class NormalIntegrator : public Integrator {
public:
    NormalIntegrator(const PropertyList& props) {}
    /// Compute the radiance value for a given ray. Just return green here
    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        /* Find the surface that is visible in the requested direction */
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        /* Return the component-wise absolute
           value of the shading normal as a color */
        Normal3f n = its.shFrame.n;
        return Color3f(std::abs(n.x()), std::abs(n.y()), std::abs(n.z()));
    }

    /// Return a human-readable description for debugging purposes
    std::string toString() const { return "NormalIntegrator[]"; }

    optix::Integrator getOptixIntegrator() const {
        optix::Integrator integrator;
        integrator.type = optix::Integrator::NORMAL;
        return integrator;
    }

protected:
    std::string m_myProperty;
};

REGISTER_CLASS(NormalIntegrator, "normals");

}  // namespace drawlab