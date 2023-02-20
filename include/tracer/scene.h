#pragma once

#include "core/base/common.h"
#include "core/parser/object.h"
#include "core/parser/proplist.h"
#include "tracer/octree.h"
#include "tracer/camera.h"
#include "tracer/emitter.h"
#include "tracer/integrator.h"
#include "tracer/mesh.h"
#include "tracer/sampler.h"
#include <set>

namespace drawlab {

/**
 * \brief Main scene data structure
 *
 * This class holds information on scene objects and is responsible for
 * coordinating rendering jobs. It also provides useful query routines that
 * are mostly used by the \ref Integrator implementations.
 */
class Scene : public Object {
public:
    Scene(const PropertyList&);

    virtual ~Scene();

    EClassType getClassType() const { return EScene; }

    /// Add a child object to the scene (meshes, integrators etc.)
    void addChild(Object* child);

    /**
     * Initializes the internal data structures (kd-tree,
     * emitter sampling data structures, etc.)
     */
    void activate();

    /// Return a string summary of the scene (for debugging purposes)
    std::string toString() const;

    /**
     * \brief Sample a point on emitters uniformly.
     *
     */
    void sampleEmitterDirection(const Intersection& its, const Point2f& sample,
                                DirectionSample& ds, Color3f& spectrum) const;

    /**
     * \brief Compute the probability of sampling \c ds .
     *
     * \param ds A direction sample record.
     *
     * \return  A probability/density value
     */
    float pdfEmitterDirection(const DirectionSample& ds) const;

    const Accel* getAccel() const { return m_accel; }

    const Integrator* getIntegrator() const { return m_integrator; }

    Integrator* getIntegrator() { return m_integrator; }

    const Camera* getCamera() const { return m_camera; }

    const Sampler* getSampler() const { return m_sampler; }

    Sampler* getSampler() { return m_sampler; }

    const std::vector<Mesh*>& getMeshes() const { return m_meshes; }

    const std::vector<Emitter*>& getEmitters() const { return m_emitters; }

    const std::vector<int>& getMeshLightIdx() const { return m_mesh_light_idx; }

    const std::vector<int>& getMeshBsdfIdx() const { return m_mesh_bsdf_idx; }

    const std::vector<int>& getLightBsdfIdx() const { return m_light_bsdf_idx; }

    const std::vector<const BSDF*>& getBSDFs() const { return m_bsdfs; }

    const Color3f& getBgColor() const { return m_bg_color; }

    bool hasEnvironmentEmitter() const { return m_has_env; }
    /**
     * \brief Intersect a ray against all triangles stored in the scene
     * and return detailed intersection information
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information
     *
     * \param its
     *    A detailed intersection record, which will be filled by the
     *    intersection query
     *
     * \return \c true if an intersection was found
     */
    bool rayIntersect(const Ray3f& ray, Intersection& its) const {
        return m_accel->rayIntersect(ray, its, false);
    }

    /**
     * \brief Intersect a ray against all triangles stored in the scene
     * and \a only determine whether or not there is an intersection.
     *
     * This method much faster than the other ray tracing function,
     * but the performance comes at the cost of not providing any
     * additional information about the detected intersection
     * (not even its position).
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information
     *
     * \return \c true if an intersection was found
     */
    bool rayIntersect(const Ray3f& ray) const {
        Intersection its; /* Unused */
        return m_accel->rayIntersect(ray, its, true);
    }

    /// \brief Return an axis-aligned box that bounds the scene
    const BoundingBox3f& getBoundingBox() const {
        return m_accel->getBoundingBox();
    }

private:
    std::vector<Mesh*> m_meshes;
    std::vector<int> m_mesh_light_idx;
    std::vector<int> m_mesh_bsdf_idx;
    std::vector<int> m_light_bsdf_idx;
    std::vector<Emitter*> m_emitters;
    std::vector<const BSDF*> m_bsdfs;
    Integrator* m_integrator = nullptr;
    Sampler* m_sampler = nullptr;
    Camera* m_camera = nullptr;
    Accel* m_accel = nullptr;
    Color3f m_bg_color;
    bool m_has_env = false;
};

}  // namespace drawlab