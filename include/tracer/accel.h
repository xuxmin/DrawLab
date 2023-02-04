#pragma once

#include "core/math/bbox.h"
#include "tracer/mesh.h"


namespace drawlab {

/**
 * \brief Acceleration data structure for ray intersection queries
 *
 */
class Accel {
public:
    /**
     * \brief Register a triangle mesh for inclusion in the acceleration
     * data structure
     *
     * This function can only be used before \ref build() is called
     */
    void addMesh(Mesh* mesh) {
        m_meshPtrs.push_back(mesh);
        m_bbox.expandBy(mesh->getBoundingBox());
    }

    /// Return an axis-aligned box that bounds the scene
    const BoundingBox3f& getBoundingBox() const { return m_bbox; }

    /**
     * \brief Intersect a ray against all triangles stored in the scene and
     * return detailed intersection information
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum extent
     *    information
     *
     * \param its
     *    A detailed intersection record, which will be filled by the
     *    intersection query
     *
     * \param shadowRay
     *    \c true if this is a shadow ray query, i.e. a query that only aims to
     *    find out whether the ray is blocked or not without returning detailed
     *    intersection information.
     *
     * \return \c true if an intersection was found
     */
    virtual bool rayIntersect(const Ray3f& ray, Intersection& its,
                              bool shadowRay) const = 0;

    /// Build the acceleration data structure
    virtual void build() = 0;


protected:
    std::vector<Mesh*> m_meshPtrs;
    BoundingBox3f m_bbox;
};

}  // namespace drawlab