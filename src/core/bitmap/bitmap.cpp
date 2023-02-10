#include "core/bitmap/bitmap.h"
#include "core/base/common.h"
#include <ImfChannelList.h>
#include <ImfIO.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfStringAttribute.h>
#include <ImfVersion.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <spdlog/spdlog.h>

namespace drawlab {

Bitmap::~Bitmap() { m_data.clear(); }

Bitmap::Bitmap(int height, int width) : m_height(height), m_width(width) {
    m_data.resize(m_width * m_height * 3);
}

Bitmap::Bitmap(const Bitmap& bitmap) {
    m_height = bitmap.getHeight();
    m_width = bitmap.getWidth();
    m_data.assign(bitmap.m_data.begin(), bitmap.m_data.end());
}

Bitmap::Bitmap(const std::string& filename) {
    filesystem::path path = getFileResolver()->resolve(filename);

    if (path.extension() == "exr") {
        loadEXR(path.str());
    }
    else {
        loadLDR(path.str());
    }
}

void Bitmap::loadEXR(const std::string& filename) {
    Imf::InputFile file(filename.c_str());
    const Imf::Header& header = file.header();
    const Imf::ChannelList& channels = header.channels();

    Imath::Box2i dw = file.header().dataWindow();
    m_width = dw.max.x - dw.min.x + 1;
    m_height = dw.max.y - dw.min.y + 1;

    m_data.resize(m_width * m_height * 3);

    spdlog::info("Reading a {}x{} OpenEXR file from \"{}\"", getHeight(),
                 getWidth(), filename);

    const char *ch_r = nullptr, *ch_g = nullptr, *ch_b = nullptr;
    for (Imf::ChannelList::ConstIterator it = channels.begin();
         it != channels.end(); ++it) {
        std::string name = toLower(it.name());

        if (it.channel().xSampling != 1 || it.channel().ySampling != 1) {
            /* Sub-sampled layers are not supported */
            continue;
        }

        if (!ch_r && (name == "r" || name == "red" || endsWith(name, ".r") ||
                      endsWith(name, ".red"))) {
            ch_r = it.name();
        } else if (!ch_g &&
                   (name == "g" || name == "green" || endsWith(name, ".g") ||
                    endsWith(name, ".green"))) {
            ch_g = it.name();
        } else if (!ch_b && (name == "b" || name == "blue" ||
                             endsWith(name, ".b") || endsWith(name, ".blue"))) {
            ch_b = it.name();
        }
    }
    if (!ch_r || !ch_g || !ch_b)
        throw Exception("This is not a standard RGB OpenEXR file!");
    size_t compStride = sizeof(float), pixelStride = 3 * compStride,
           rowStride = pixelStride * getWidth();

    char* ptr = reinterpret_cast<char*>(&m_data[0]);

    Imf::FrameBuffer frameBuffer;
    frameBuffer.insert(ch_r,
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));
    ptr += compStride;
    frameBuffer.insert(ch_g,
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));
    ptr += compStride;
    frameBuffer.insert(ch_b,
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));
    file.setFrameBuffer(frameBuffer);
    file.readPixels(dw.min.y, dw.max.y);
}

void Bitmap::loadLDR(const std::string& filename) {
    
    stbi_set_flip_vertically_on_load(true);

    int w, h, c;
    int ok = stbi_info(filename.c_str(), &w, &h, &c);
    if (!ok) {
        std::cerr << "can't load file " << filename << "\n";
        return;
    }
    unsigned char* tmp_data = stbi_load((filename).c_str(), &w, &h, &c, 0);

    m_data.resize(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        if (c == 1) {
            m_data[i] = tmp_data[i / 3];
        }
        else {
            m_data[i] = tmp_data[i];
        }
    }

    m_width = w;
    m_height = h;
}

void Bitmap::saveEXR(const std::string& filename) {
    spdlog::info("Writing a {}x{} OpenEXR file to \"{}.exr\"", getHeight(),
                 getWidth(), filename);

    std::string path = filename + ".exr";

    Imf::Header header((int)getWidth(), (int)getHeight());
    header.insert("comments", Imf::StringAttribute("Add comments here!"));

    Imf::ChannelList& channels = header.channels();
    channels.insert("R", Imf::Channel(Imf::FLOAT));
    channels.insert("G", Imf::Channel(Imf::FLOAT));
    channels.insert("B", Imf::Channel(Imf::FLOAT));

    Imf::FrameBuffer frameBuffer;
    size_t compStride = sizeof(float), pixelStride = 3 * compStride,
           rowStride = pixelStride * getWidth();

    char* ptr = reinterpret_cast<char*>(getPtr());
    frameBuffer.insert("R",
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));
    ptr += compStride;
    frameBuffer.insert("G",
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));
    ptr += compStride;
    frameBuffer.insert("B",
                       Imf::Slice(Imf::FLOAT, ptr, pixelStride, rowStride));

    Imf::OutputFile file(path.c_str(), header);
    file.setFrameBuffer(frameBuffer);
    file.writePixels((int)getHeight());
}

void Bitmap::savePNG(const std::string& filename) {
    spdlog::info("Writing a {}x{} PNG file to \"{}.png\"", getHeight(),
                 getWidth(), filename);

    std::string path = filename + ".png";

    uint8_t* rgb8 = new uint8_t[3 * getWidth() * getHeight()];

    for (int i = 0; i < getHeight(); i++) {
        for (int j = 0; j < getWidth(); j++) {
            int idx = (i * getWidth() + j) * 3;
            Color3f tonemapped = getPixel(i, j).toSRGB();
            rgb8[idx] = (uint8_t)clamp(255.f * tonemapped[0], 0.f, 255.f);
            rgb8[idx + 1] = (uint8_t)clamp(255.f * tonemapped[1], 0.f, 255.f);
            rgb8[idx + 2] = (uint8_t)clamp(255.f * tonemapped[2], 0.f, 255.f);
        }
    }

    int ret = stbi_write_png(path.c_str(), getWidth(), getHeight(), 3, rgb8,
                             3 * getWidth());
    if (ret == 0) {
        spdlog::error("Bitmap::savePNG(): Could not save PNG file \"{}\"", path);
    }
    delete[] rgb8;
}

Color3f Bitmap::getPixel(int row, int col) const {
    if (row < 0 || col < 0 || row > getHeight() || col > getWidth()) {
        return Color3f();
    }

    int p = (row * getWidth() + col) * 3;
    Color3f color = Color3f(m_data[p], m_data[p + 1], m_data[p + 2]);
    return color;
}

void Bitmap::setPixel(int row, int col, Color3f color) {
    if (row < 0 || col < 0 || row > getHeight() || col > getWidth()) {
        return;
    }

    int p = (row * getWidth() + col) * 3;
    m_data[p] = color[0];
    m_data[p + 1] = color[1];
    m_data[p + 2] = color[2];
}

void Bitmap::flipud() {
    for (int h = 0; h < getHeight() / 2; h++) {
        int p0 = h * getWidth() * 3;
        int p1 = (h + 1) * getWidth() * 3;
        int p2 = (getHeight() - h - 1) * getWidth() * 3;
        std::swap_ranges(m_data.begin() + p0, m_data.begin() + p1,
                         m_data.begin() + p2);
    }
}

void Bitmap::resize(int new_height, int new_width) {
    std::vector<float> data_tmp(m_data);
    m_data.resize(new_height * new_width * 3);

    for (int h = 0; h < new_height; h++) {
        for (int w = 0; w < new_width; w++) {
            int ori_h = (int)((float)h / (float)new_height * m_height);
            int ori_w = (int)((float)w / (float)new_width * m_width);

            ori_h = std::min(ori_h, m_height - 1);
            ori_w = std::min(ori_w, m_width - 1);

            int ori_p = (ori_h * m_width + ori_w) * 3;
            int new_p = (h * new_width + w) * 3;
            m_data[new_p] = data_tmp[ori_p];
            m_data[new_p + 1] = data_tmp[ori_p + 1];
            m_data[new_p + 2] = data_tmp[ori_p + 2];
        }
    }
    m_height = new_height;
    m_width = new_width;
}

void Bitmap::colorNan(Color3f invalid_color) {
    for (int i = 0; i < m_height; i++) {
        for (int j = 0; j < m_width; j++) {
            if (!getPixel(i, j).isValid()) {
                setPixel(i, j, invalid_color);
                spdlog::warn("There is invalid pixel ({}, {}) in the bitmap!", j, i);
            }
        }
    }
}

}  // namespace drawlab