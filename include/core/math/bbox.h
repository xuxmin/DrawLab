#pragma once

#include "core/base/common.h"
#include "core/math/ray.h"

namespace drawlab {

/**
 * \brief Generic n-dimensional bounding box data structure
 *
 * Maintains a minimum and maximum position along each dimension and provides
 * various convenience functions for querying and modifying them.
 *
 * This class is parameterized by the underlying point data structure,
 * which permits the use of different scalar types and dimensionalities, e.g.
 * \code
 * TBoundingBox<Vector3i> integerBBox(Point3i(0, 1, 3), Point3i(4, 5, 6));
 * TBoundingBox<Vector2d> doubleBBox(Point2d(0.0, 1.0), Point2d(4.0, 5.0));
 * \endcode
 *
 * \tparam T The underlying point data type (e.g. \c Point2d)
 * \ingroup libcore
 */

template <size_t N, typename T> struct TBoundingBox {
    typedef TPoint<N, T> PointType;
    typedef TVector<N, T> VectorType;

    /**
     * \brief Create a new invalid bounding box
     *
     * Initializes the components of the minimum
     * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
     * respectively.
     */
    TBoundingBox() { reset(); }

    /// @brief Create a collapsed bounding box from a single point
    TBoundingBox(const PointType& p) : m_min(p), m_max(p) {}

    /// @brief Create a bounding box from two positions
    TBoundingBox(const PointType& min, const PointType& max)
        : m_min(min), m_max(max) {}

    /// @brief Test for equality against another bounding box
    bool operator==(const TBoundingBox& bbox) const {
        return m_min == bbox.m_min && m_max == bbox.m_max;
    }

    /// @brief Test for inequality against another bounding box
    bool operator!=(const TBoundingBox& bbox) const {
        return m_min != bbox.m_min || m_max != bbox.m_max;
    }

    /// @brief Calculate the n-dimensional volume of the bounding box
    T getVolume() const { return (m_max - m_min).prod(); }

    /// @brief Calculate the n-1 dimensional volume of the boundary
    T getSurfaceArea() const {
        VectorType d = m_max - m_min;
        T result = 0.0f;
        for (size_t i = 0; i < N; i++) {
            T term = 1.0f;
            for (size_t j = 0; j < N; j++) {
                if (i == j)
                    continue;
                term *= d[j];
            }
            result += term;
        }
        return 2.f * result;
    }

    /// @brief Return the center point
    PointType getCenter() const { return (m_max + m_min) * (T)0.5f; }

    /**
     * \brief Check whether a point lies \a on or \a inside the bounding box
     *
     * \param p The point to be tested
     *
     * \param strict Set this parameter to \c true if the bounding
     *               box boundary should be excluded in the test
     */
    bool contains(const PointType& p, bool strict = false) const {
        if (strict) {
            return (p > m_min).all() && (p < m_max).all();
        } else {
            return (p >= m_min).all() && (p <= m_max).all();
        }
    }

    /**
     * \brief Check whether a specified bounding box lies \a on or \a within
     * the current bounding box
     *
     * Note that by definition, an 'invalid' bounding box (where
     * min=\f$\infty\f$ and max=\f$-\infty\f$) does not cover any space. Hence,
     * this method will always return \a true when given such an argument.
     *
     * \param strict Set this parameter to \c true if the bounding
     *               box boundary should be excluded in the test
     */
    bool contains(const TBoundingBox& bbox, bool strict = false) const {
        if (strict) {
            return (bbox.m_min > m_min).all() && (bbox.m_max < m_max).all();
        } else {
            return (bbox.m_min >= m_min).all() && (bbox.m_max <= m_max).all();
        }
    }

    /**
     * \brief Check two axis-aligned bounding boxes for possible overlap.
     *
     * \param strict Set this parameter to \c true if the bounding
     *               box boundary should be excluded in the test
     *
     * \return \c true If overlap was detected.
     */
    bool overlaps(const TBoundingBox& bbox, bool strict = false) const {
        if (strict) {
            return (bbox.m_min < m_max).all() && (bbox.m_max > m_min).all();
        } else {
            return (bbox.m_min <= m_max).all() && (bbox.m_max >= m_min).all();
        }
    }

    /**
     * \brief Mark the bounding box as invalid.
     *
     * This operation sets the components of the minimum
     * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
     * respectively.
     */
    void reset() {
        m_min.setConstant(std::numeric_limits<T>::infinity());
        m_max.setConstant(-std::numeric_limits<T>::infinity());
    }

    /// Expand the bounding box to contain another point
    void expandBy(const PointType& p) {
        m_min = m_min.cwiseMin(p);
        m_max = m_max.cwiseMax(p);
    }

    /// Expand the bounding box to contain another bounding box
    void expandBy(const TBoundingBox& bbox) {
        m_min = m_min.cwiseMin(bbox.m_min);
        m_max = m_max.cwiseMax(bbox.m_max);
    }

    /// Clip to another bounding box
    void clip(const TBoundingBox& bbox) {
        m_min = m_min.cwiseMax(bbox.m_min);
        m_max = m_max.cwiseMin(bbox.m_max);
    }

    /**
     * \brief Calculate the smallest square distance between
     * the axis-aligned bounding box and \c bbox.
     */
    T squaredDistanceTo(const TBoundingBox& bbox) const {
        T result = 0;

        for (int i = 0; i < N; ++i) {
            T value = 0;
            if (bbox.m_max[i] < m_min[i])
                value = m_min[i] - bbox.m_max[i];
            else if (bbox.m_min[i] > m_max[i])
                value = bbox.m_min[i] - m_max[i];
            result += value * value;
        }

        return result;
    }

    /**
     * \brief Calculate the smallest distance between
     * the axis-aligned bounding box and the point \c p.
     */
    T distanceTo(const PointType& p) const {
        return std::sqrt(squaredDistanceTo(p));
    }

    /**
     * \brief Check whether this is a valid bounding box
     *
     * A bounding box \c bbox is valid when
     * \code
     * bbox.min[dim] <= bbox.max[dim]
     * \endcode
     * holds along each dimension \c dim.
     */
    bool isValid() const { return (m_max >= m_min).all(); }

    /// Check whether this bounding box has collapsed to a single point
    bool isPoint() const { return (m_max == m_min).all(); }

    /// Check whether this bounding box has any associated volume
    bool hasVolume() const { return (m_max > m_min).all(); }

    /// Return the dimension index with the largest associated side length
    int getMajorAxis() const {
        VectorType d = m_max - m_min;
        int largest = 0;
        for (int i = 1; i < N; ++i)
            if (d[i] > d[largest])
                largest = i;
        return largest;
    }

    /// Return the dimension index with the shortest associated side length
    int getMinorAxis() const {
        VectorType d = m_max - m_min;
        int shortest = 0;
        for (int i = 1; i < N; ++i)
            if (d[i] < d[shortest])
                shortest = i;
        return shortest;
    }

    /**
     * \brief Calculate the bounding box extents
     * \return max-min
     */
    VectorType getExtents() const { return m_max - m_min; }

    /// Return a string representation of the bounding box
    std::string toString() const {
        // if (!isValid())
        //     return "BoundingBox[invalid]";
        // else
        return tfm::format("BoundingBox[min=%s, max=%s]", m_min.toString(),
                           m_max.toString());
    }

    /// Return the index of the largest axis
    int getLargestAxis() const {
        VectorType extents = m_max - m_min;

        if (extents[0] >= extents[1] && extents[0] >= extents[2])
            return 0;
        else if (extents[1] >= extents[0] && extents[1] >= extents[2])
            return 1;
        else
            return 2;
    }

    /// Return the position of a bounding box corner
    PointType getCorner(int index) const {
        PointType result;
        for (int i = 0; i < N; ++i)
            result[i] = (index & (1 << i)) ? m_max[i] : m_min[i];
        return result;
    }

    /// Check if a ray intersects a bounding box
    bool rayIntersect(const Ray3f& ray) const {
        float nearT = -std::numeric_limits<float>::infinity();
        float farT = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; i++) {
            float origin = ray.o[i];
            float minVal = m_min[i], maxVal = m_max[i];

            if (ray.d[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
            } else {
                float t1 = (minVal - origin) * ray.dRcp[i];
                float t2 = (maxVal - origin) * ray.dRcp[i];

                if (t1 > t2)
                    std::swap(t1, t2);

                nearT = std::max(t1, nearT);
                farT = std::min(t2, farT);

                if (!(nearT <= farT))
                    return false;
            }
        }

        return ray.mint <= farT && nearT <= ray.maxt;
    }

    /// Return the overlapping region of the bounding box and an unbounded ray
    bool rayIntersect(const Ray3f& ray, float& nearT, float& farT) const {
        nearT = -std::numeric_limits<float>::infinity();
        farT = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; i++) {
            float origin = ray.o[i];
            float minVal = m_min[i], maxVal = m_max[i];

            if (ray.d[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
            } else {
                float t1 = (minVal - origin) * ray.dRcp[i];
                float t2 = (maxVal - origin) * ray.dRcp[i];

                if (t1 > t2)
                    std::swap(t1, t2);

                nearT = std::max(t1, nearT);
                farT = std::min(t2, farT);

                if (!(nearT <= farT))
                    return false;
            }
        }

        return true;
    }

    PointType m_min;  /// Component-wise minimum
    PointType m_max;  /// Component-wise maximum
};

};  // namespace drawlab
