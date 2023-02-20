#pragma once

#include "tracer/mesh.h"

namespace drawlab {

class Emitter : public Object {
public:
    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    EClassType getClassType() const { return EEmitter; }

    /// Return the shape, to which the emitter is currently attached
    Mesh* getMesh() { return m_mesh; }

    /// Return the shape, to which the emitter is currently attached (const
    /// version)
    const Mesh* getMesh() const { return m_mesh; }

    /// Set the shape associated with this emitter.
    void setMesh(Mesh* mesh) {
        if (m_mesh)
            throw("An Emitter can be only be attached to a single mesh.");

        m_mesh = mesh;
    }

    virtual bool isEnvironmentEmitter() const { return false; }

    /**
     * \brief Given a reference point in the scene, sample a direction from the
     * reference point towards the Emitter (ideally proportional to the
     * emission/sensitivity profile)
     *
     * This operation is a generalization of direct illumination techniques to
     * both emitters \a and sensors. A direction sampling method is given an
     * arbitrary reference position in the scene and samples a direction from
     * the reference point towards the endpoint (ideally proportional to the
     * emission/sensitivity profile). This reduces the sampling domain from 4D
     * to 2D, which often enables the construction of smarter specialized
     * sampling techniques.
     *
     * Ideally, the implementation should importance sample the product of
     * the emission profile and the geometry term between the reference point
     * and the position on the endpoint.
     *
     * That is:
     * - lp is the point on light, ln is the light normal
     * - xp is the point on surface, xn is the surface normal
     * - d is the vector from surface point to light point
     * - Le is the light illumination
     * - pA is the probability relative to light area
     * - pw is the probability relative to solid angle
     *
     *   pA = pw * |cos(-d·ln)| / ||lp - xp||^2
     *
     *   direct lumi = (Le * f(wi, wo) * G(lp <-> xp)) / pA
     *               = (Le * f(wi, wo) * V(lp <-> xp) * |cos(d·xn)| *
     * |cos(-d·ln)| / ||lp - xp||^2) / pA = (Le * f(wi, wo) * V(lp <-> xp) *
     * |cos(d·xn)|) / pw spectrum = (Le * |cos(d·xn)|) / pw
     *
     * The default implementation throws an exception.
     *
     * \param ref
     *    A reference position somewhere within the scene.
     * \param ds
     *     A \ref DirectionSample instance describing the generated sample.
     * \param spectrum
     *     A spectral importance weight
     */
    virtual void sampleDirection(const Point2f& sample, const Intersection& its,
                                 DirectionSample& ds, Color3f& spectrum) const {
        throw Exception("Emitter::sampleDirection(): not implemented!");
    }

    /**
     * \brief Query the probability density of \ref sample_direction()
     *
     * \param it
     *    A reference position somewhere within the scene.
     *
     * \param ps
     *     A position record describing the sample in question
     *
     * \return
     *     The probability density per unit solid angle
     */
    virtual float pdfDirection(const DirectionSample& ds) const {
        throw Exception("Emitter::pdfDirection(): not implemented!");
    }

    /**
     * \brief Given a ray-surface intersection, return the emitted
     * radiance or importance traveling along the reverse direction
     *
     * This function is e.g. used when an area light source has been hit by a
     * ray in a path tracing-style integrator, and it subsequently needs to be
     * queried for the emitted radiance along the negative ray direction. The
     * default implementation throws an exception, which states that the method
     * is not implemented.
     *
     * \param its
     *    An intersect record that specfies both the query position
     *    and direction (using the <tt>si.wi</tt> field)
     * \return
     *    The emitted radiance or importance
     */
    virtual Color3f eval(const Intersection& its, const Vector3f wi) const {
        throw Exception("Emitter::eval(): not implemented!");
    }

    virtual void getOptixLight(optix::Light& light) const = 0;

protected:
    Mesh* m_mesh = nullptr;
};

}  // namespace drawlab