#pragma once

#include "core/bitmap/color.h"
#include <vector>

namespace drawlab {

/**
 * \brief Store a RGB high dynamic-range bitmap
 */
class Bitmap {
public:
    /// Create an empty image with size (height, width, 3)
    Bitmap(int height, int width);

    /// Load an OpenEXR file with the specified filename
    Bitmap(const std::string& filename);

    /// Release memory
    ~Bitmap();

    /// Copy a bitmap
    Bitmap(const Bitmap& bitmap);

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    float* getPtr() { return &m_data[0]; }

    /// @brief Get pixel data in [row, col]
    Color3f getPixel(int row, int col) const;

    void setPixel(int row, int col, Color3f color);

    void saveEXR(const std::string& filename);
    void savePNG(const std::string& filename);

    void flipud();

    /// Simply resize image
    void resize(int new_height, int new_width);

    // Test any invalid pixel
    void colorNan(Color3f invalid_color);

private:
    int m_width;
    int m_height;
    std::vector<float> m_data;

    void loadEXR(const std::string& filename);
    void loadLDR(const std::string& filename);
};

}  // namespace drawlab
