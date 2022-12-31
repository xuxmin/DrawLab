
/* =======================================================================
     This file contains classes for parallel rendering of "image blocks".
 * ======================================================================= */

#pragma once

#include "core/base/common.h"
#include "core/bitmap/color.h"
#include "core/math/point.h"
#include "core/math/vector.h"
#include <tbb/mutex.h>

namespace drawlab {

/**
 * \brief Weighted pixel storage for a rectangular subregion of an image
 *
 * This class implements storage for a rectangular subregion of a
 * larger image that is being rendered. For each pixel, it records color
 * values along with a weight that specifies the accumulated influence of
 * nearby samples on the pixel (according to the used reconstruction filter).
 *
 * When rendering with filters, the samples in a rectangular
 * region will generally also contribute to pixels just outside of
 * this region. For that reason, this class also stores information about
 * a small border region around the rectangle, whose size depends on the
 * properties of the reconstruction filter.
 */
class ImageBlock {
public:
    /**
     * Create a new image block of the specified maximum size
     * \param size
     *     Desired maximum size of the block
     * \param filter
     *     Samples will be convolved with the image reconstruction
     *     filter provided here.
     */
    ImageBlock(const Vector2i& size, const ReconstructionFilter* filter);

    /// Create a new image block from the bitmap
    ImageBlock(const Bitmap& bitmap);

    /// Release all memory
    ~ImageBlock();

    /// Configure the offset of the block within the main image
    void setOffset(const Point2i& offset) { m_offset = offset; }

    /// Return the offset of the block within the main image
    inline const Point2i& getOffset() const { return m_offset; }

    /// Configure the size of the block within the main image
    void setSize(const Vector2i& size) { m_size = size; }

    /// Return the size of the block within the main image, [width, height]
    inline const Vector2i& getSize() const { return m_size; }

    inline const int cols() const { return m_size[0] + 2 * m_borderSize; }
    inline const int rows() const { return m_size[1] + 2 * m_borderSize; }

    Color4f& coeffRef(int row, int col);
    const Color4f& coeffRef(int row, int col) const;

    /// Return the border size in pixels
    inline int getBorderSize() const { return m_borderSize; }

    /**
     * \brief Turn the block into a proper bitmap
     *
     * This entails normalizing all pixels and discarding
     * the border region.
     */
    Bitmap* toBitmap() const;

    /// Convert a bitmap into an image block
    void fromBitmap(const Bitmap& bitmap);

    /// Clear all contents
    void clear() {
        for (int i = 0; i < m_data.size(); i++) {
            m_data[i] = Color4f();
        }
    }

    /// Record a sample with the given position and radiance value
    void put(const Point2f& pos, const Color3f& value);

    /**
     * \brief Merge another image block into this one
     *
     * During the merge operation, this function locks
     * the destination block using a mutex.
     */
    void put(ImageBlock& b);

    /// Lock the image block (using an internal mutex)
    inline void lock() const { m_mutex.lock(); }

    /// Unlock the image block
    inline void unlock() const { m_mutex.unlock(); }

    /// Return a human-readable string summary
    std::string toString() const;

protected:
    std::vector<Color4f> m_data;

    Point2i m_offset;
    Vector2i m_size;  // [width, height]
    int m_borderSize = 0;
    float* m_filter = nullptr;
    float m_filterRadius = 0;
    float* m_weightsX = nullptr;
    float* m_weightsY = nullptr;
    float m_lookupFactor = 0;
    mutable tbb::mutex m_mutex;
};

}  // namespace drawlab