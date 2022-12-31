#include "editor/block.h"
#include "core/bitmap/bitmap.h"
#include "core/math/bbox.h"
#include "tracer/rfilter.h"
#include <tbb/tbb.h>

namespace drawlab {

ImageBlock::ImageBlock(const Vector2i& size, const ReconstructionFilter* filter)
    : m_offset(Point2i(0, 0)), m_size(size) {
    if (filter) {
        /* Tabulate the image reconstruction filter for performance reasons */
        m_filterRadius = filter->getRadius();
        m_borderSize = (int)std::ceil(m_filterRadius - 0.5f);
        m_filter = new float[FILTER_RESOLUTION + 1];
        for (int i = 0; i < FILTER_RESOLUTION; ++i) {
            float pos = (m_filterRadius * i) / FILTER_RESOLUTION;
            m_filter[i] = filter->eval(pos);
        }
        m_filter[FILTER_RESOLUTION] = 0.0f;
        m_lookupFactor = FILTER_RESOLUTION / m_filterRadius;
        int weightSize = (int)std::ceil(2 * m_filterRadius) + 1;
        m_weightsX = new float[weightSize];
        m_weightsY = new float[weightSize];
        memset(m_weightsX, 0, sizeof(float) * weightSize);
        memset(m_weightsY, 0, sizeof(float) * weightSize);
    }

    /* Allocate space for pixels and border regions */
    m_data.resize((size.x() + 2 * m_borderSize) *
                  (size.y() + 2 * m_borderSize));
}

ImageBlock::ImageBlock(const Bitmap& bitmap) {
    m_borderSize = 0;
    m_size = Vector2i(bitmap.getWidth(), bitmap.getHeight());
    m_data.resize(bitmap.getHeight() * bitmap.getWidth());
    fromBitmap(bitmap);
}

ImageBlock::~ImageBlock() {
    delete[] m_filter;
    delete[] m_weightsX;
    delete[] m_weightsY;
}

Color4f& ImageBlock::coeffRef(int row, int col) {
    int p = row * cols() + col;
    return m_data[p];
}

const Color4f& ImageBlock::coeffRef(int row, int col) const {
    int p = row * cols() + col;
    return m_data[p];
}

Bitmap* ImageBlock::toBitmap() const {
    Bitmap* result = new Bitmap(m_size.y(), m_size.x());
    for (int y = 0; y < m_size.y(); ++y) {
        for (int x = 0; x < m_size.x(); ++x) {
            result->setPixel(y, x, coeffRef(y, x).divideByFilterWeight());
        }
    }
    return result;
}

void ImageBlock::fromBitmap(const Bitmap& bitmap) {
    if (bitmap.getHeight() != m_size[1] + m_borderSize * 2 ||
        bitmap.getWidth() != m_size[0] + m_borderSize * 2) {
        throw Exception("Invalid bitmap dimensions!");
    }

    for (int y = 0; y < m_size.y(); ++y) {
        for (int x = 0; x < m_size.x(); ++x) {
            int p = y * (m_borderSize + m_size.x()) + x + m_borderSize;
            coeffRef(y, x) = Color4f(bitmap.getPixel(y, x));
        }
    }
}

void ImageBlock::put(const Point2f& _pos, const Color3f& value) {
    if (!value.isValid()) {
        /* If this happens, go fix your code instead of removing this warning ;)
         */
        cerr << "Integrator: computed an invalid radiance value: "
             << value.toString() << endl;
        return;
    }

    /* Convert to pixel coordinates within the image block */
    Point2f pos(_pos.x() - 0.5f - (m_offset.x() - m_borderSize),
                _pos.y() - 0.5f - (m_offset.y() - m_borderSize));

    /* Compute the rectangle of pixels that will need to be updated */
    BoundingBox2i bbox(Point2i((int)std::ceil(pos.x() - m_filterRadius),
                               (int)std::ceil(pos.y() - m_filterRadius)),
                       Point2i((int)std::floor(pos.x() + m_filterRadius),
                               (int)std::floor(pos.y() + m_filterRadius)));
    bbox.clip(BoundingBox2i(Point2i(0, 0),
                            Point2i((int)cols() - 1, (int)rows() - 1)));

    /* Lookup values from the pre-rasterized filter */
    for (int x = bbox.m_min.x(), idx = 0; x <= bbox.m_max.x(); ++x)
        m_weightsX[idx++] =
            m_filter[(int)(std::abs(x - pos.x()) * m_lookupFactor)];
    for (int y = bbox.m_min.y(), idx = 0; y <= bbox.m_max.y(); ++y)
        m_weightsY[idx++] =
            m_filter[(int)(std::abs(y - pos.y()) * m_lookupFactor)];

    for (int y = bbox.m_min.y(), yr = 0; y <= bbox.m_max.y(); ++y, ++yr)
        for (int x = bbox.m_min.x(), xr = 0; x <= bbox.m_max.x(); ++x, ++xr)
            coeffRef(y, x) += Color4f(value) * m_weightsX[xr] * m_weightsY[yr];
}

}  // namespace drawlab