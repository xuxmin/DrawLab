
#include "optix/optix_renderer.h"
#include "optix/sutil.h"
#include <exception>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace sutil;

namespace optix {

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char** av) {
    try {
        OptixRenderer renderer;
        const int width = 1200, height = 1024;
        renderer.resize(width, height);
        renderer.render();

        std::vector<float> pixels(width * height);
        renderer.downloadPixels(pixels.data());

        const std::string fileName = "osc_example2.png";
        stbi_write_png(fileName.c_str(), width, height, 4, pixels.data(),
                       width * sizeof(float));
        std::cout << TERMINAL_GREEN << std::endl
                  << "Image rendered, and saved to " << fileName << " ... done."
                  << std::endl
                  << TERMINAL_DEFAULT << std::endl;

    } catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}

}  // namespace optix