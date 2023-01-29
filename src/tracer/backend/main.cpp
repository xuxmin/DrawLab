
#include "tracer/backend/optix_renderer.h"
#include "optix/sutil.h"
#include "tracer/mesh.h"
#include <exception>
#include <iostream>
#include "stb_image_write.h"
#include "core/base/common.h"
#include "core/parser/parser.h"
#include <spdlog/spdlog.h>

using namespace drawlab;
using namespace optix;


std::shared_ptr<Object> load_xml(const char* xml_file) {
    std::string sceneName = "";
    filesystem::path path(xml_file);

    try {
        if (path.extension() == "xml") {
            sceneName = xml_file;
            getFileResolver()->prepend(path.parent_path());
            std::shared_ptr<Object> root(loadFromXML(sceneName));
            return root;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return nullptr;
    }
    return nullptr;
}

/*! main entry point to this example - initially optix, print hello
  world, then exit */
int main(int argc, char** argv) {

    std::shared_ptr<Object> root = load_xml(argv[1]);

    try {
        OptixRenderer renderer(static_cast<Scene*>(root.get()));
        const int width = 768, height = 512;
        renderer.resize(height, width);
        renderer.updateCamera();
        renderer.render();

        std::vector<unsigned int> pixels(width * height);
        renderer.downloadPixels(pixels.data());

        const std::string fileName = "osc_example2.png";
        stbi_write_png(fileName.c_str(), width, height, 4, pixels.data(),
                       width * sizeof(unsigned int));
        spdlog::info("Image rendered, and saved to {}... done.", fileName);

    } catch (std::runtime_error& e) {
        spdlog::critical("FATAL ERROR: {}", e.what());
        exit(1);
    }
    return 0;
}
