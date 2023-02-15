#pragma once

#include "core/math/bbox.h"
#include "core/math/dpdf.h"
#include "core/math/frame.h"
#include "core/parser/object.h"
#include "tracer/bsdf.h"

namespace drawlab {

/**
 * \brief Intersection data structure
 *
 * This data structure records local information about a ray-triangle
 * intersection. This includes the position, traveled ray distance, uv
 * coordinates, as well as well as two local coordinate frames (one that
 * corresponds to the true geometry, and one that is used for shading
 * computations).
 */
struct Intersection {
    /// Position of the surface intersection
    Point3f p;
    /// Unoccluded distance along the ray
    float t;
    /// UV coordinates, if any
    Point2f uv;
    /// Shading frame (based on the shading normal)
    Frame shFrame;
    /// Geometric frame (based on the true geometry)
    Frame geoFrame;
    /// Pointer to the associated mesh
    const Mesh* mesh;

    /// Create an uninitialized intersection record
    Intersection() : mesh(nullptr) {}

    Intersection(Intersection&& its)
        : p(its.p), t(its.t), uv(its.uv), shFrame(its.shFrame),
          geoFrame(its.geoFrame), mesh(its.mesh) {
        its.mesh = nullptr;
    }

    Intersection(const Intersection& its)
        : p(its.p), t(its.t), uv(its.uv), shFrame(its.shFrame),
          geoFrame(its.geoFrame), mesh(its.mesh) {}

    Intersection& operator=(Intersection&& its) {
        if (this != &its) {
            this->p = its.p;
            this->t = its.t;
            this->uv = its.uv;
            this->shFrame = its.shFrame;
            this->geoFrame = its.geoFrame;
            this->mesh = its.mesh;
            its.mesh = nullptr;
        }
        return *this;
    }

    /// Transform a direction vector into the local shading frame
    Vector3f toLocal(const Vector3f& d) const { return shFrame.toLocal(d); }

    /// Transform a direction vector from local to world coordinates
    Vector3f toWorld(const Vector3f& d) const { return shFrame.toWorld(d); }

    /// Return a human-readable summary of the intersection record
    std::string toString() const;
};

/**
 * \brief Direction data structure
 *
 * This data structure records direction information about the sampled
 * direction.
 */
struct DirectionSample {
    /// Direction origin
    Point3f o;
    /// Normalized direction
    Vector3f d;
    // Normal in dst
    Normal3f n;

    const Object* obj;

    float dist;
    float pdf;

    DirectionSample() : obj(nullptr) {}

    DirectionSample(const Point3f& ori, const Point3f& dst, const Normal3f n)
        : o(ori), n(n), obj(nullptr) {
        Vector3f vec = dst - ori;
        d = vec.normalized();
        dist = vec.length();
    }

    DirectionSample(const Point3f& ori, const Point3f& dst, const Normal3f n,
                    const Object* obj)
        : o(ori), n(n), obj(obj) {
        Vector3f vec = dst - ori;
        d = vec.normalized();
        dist = vec.length();
    }
};

/**
 * \brief Triangle mesh
 *
 * This class stores a triangle mesh object and provides numerous functions
 * for querying the individual triangles. Subclasses of \c Mesh implement
 * the specifics of how to create its contents (e.g. by loading from an
 * external file)
 */
class Mesh : public Object {
public:
    /// Release all memory
    virtual ~Mesh();

    /// Initialize internal data structures (called once by the XML parser)
    virtual void activate();

    /// Register a child object (e.g. a BSDF) with the mesh
    virtual void addChild(Object* child);

    /// Return the name of this mesh
    const std::string& getName() const { return m_name; }

    /// Return a human-readable summary of this instance
    std::string toString() const;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    EClassType getClassType() const { return EMesh; }

    //// Return an axis-aligned bounding box of the entire mesh
    const BoundingBox3f& getBoundingBox() const { return m_bbox; }

    //// Return an axis-aligned bounding box containing the given triangle
    BoundingBox3f getBoundingBox(uint32_t index) const;

    /// Is this mesh an area emitter?
    bool isEmitter() const { return m_emitter != nullptr; }

    /// Return a pointer to an attached area emitter instance
    Emitter* getEmitter() { return m_emitter; }

    /// Return a pointer to an attached area emitter instance (const version)
    const Emitter* getEmitter() const { return m_emitter; }

    /// Return a pointer to the BSDF associated with this mesh
    const BSDF* getBSDF() const { return m_bsdf; }

    /// Sample a point on the surface of the mesh.
    void samplePosition(const Point2f& sample, Point3f& position,
                        Normal3f& normal) const;

    /**
     * \brief Query the probability density of \ref samplePosition() for
     * a particular point on the surface.
     */
    float pdfPosition() const;

    /// Return the total number of triangles in this shape
    uint32_t getTriangleCount() const { return (uint32_t)m_F.size() / 3; }

    /// Return the total number of vertices in this shape
    uint32_t getVertexCount() const { return (uint32_t)m_V.size() / 3; }

    /// Return the surface area of the given triangle
    float surfaceArea(uint32_t index) const;

    //// Return the centroid of the given triangle
    Point3f getCentroid(uint32_t index) const;

    /// Return the vertex texcoord of the given triangle and index
    Point2f getVertexTexCoord(size_t iface, size_t index) const;

    /// Return the vertex position of the given triangle and index
    Point3f getVertexPosition(size_t iface, size_t index) const;

    /// Return the vertex normal of the given triangle and index
    Normal3f getVertexNormal(size_t iface, size_t index) const;

    /// @brief Return the all vertex position
    const std::vector<float>& getVertexPosition() const { return m_V; }
    
    /// @brief Return the all vertex index
    const std::vector<unsigned int>& getVertexIndex() const { return m_F; }

    /// @brief Return the all vertex normal
    const std::vector<float>& getVertexNormal() const { return m_N; }

    const std::vector<float>& getVertexTexCoord() const { return m_UV; }

    /// @brief Return whether the mesh has texcoord
    bool hasTexCoord() const { return m_UV.size() > 0; }

    /// @brief Return whether the mesh has vertex normal
    bool hasVertexNormal() const { return m_N.size() > 0; }

    /** \brief Ray-triangle intersection test
     *
     * Uses the algorithm by Moeller and Trumbore discussed at
     * <tt>http://www.acm.org/jgt/papers/MollerTrumbore97/code.html</tt>.
     *
     * Note that the test only applies to a single triangle in the mesh.
     * An acceleration data structure like \ref BVH is needed to search
     * for intersections against many triangles.
     *
     * \param index
     *    Index of the triangle that should be intersected
     * \param ray
     *    The ray segment to be used for the intersection query
     * \param t
     *    Upon success, \a t contains the distance from the ray origin to the
     *    intersection point,
     * \param u
     *   Upon success, \c u will contain the 'U' component of the intersection
     *   in barycentric coordinates
     * \param v
     *   Upon success, \c v will contain the 'V' component of the intersection
     *   in barycentric coordinates
     * \return
     *   \c true if an intersection has been detected
     */
    bool rayIntersect(uint32_t index, const Ray3f& ray, float& u, float& v,
                      float& t) const;

protected:
    Mesh();

protected:
    BoundingBox3f m_bbox;
    std::string m_name;
    std::vector<float> m_V;         // Vertex positions
    std::vector<float> m_N;         // Vertex normals
    std::vector<float> m_UV;        // Vertex texture coordinates
    std::vector<unsigned int> m_F;  // Faces
    BSDF* m_bsdf = nullptr;         ///< BSDF of the surface
    Emitter* m_emitter = nullptr;   ///< Associated emitter, if any
    DiscretePDF* m_dpdf = nullptr;
};

}  // namespace drawlab