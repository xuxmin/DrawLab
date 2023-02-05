#include "cmdline.h"
#include "core/base/common.h"
#include "core/parser/parser.h"
#include "tracer/backend/cpu_renderer.h"
#include "tracer/backend/optix_renderer.h"
#include "tracer/scene.h"
#include <spdlog/spdlog.h>
#include <memory>
#include <string>

using namespace drawlab;

int main(int argc, char** argv) {
    cmdline::parser a;
    a.add<std::string>("scene", 's', "Scene xml file", true, "");
    a.add<int>("thread", 't', "Thread num used in cpu backend", false, 4);
    a.add<std::string>("backend", 'b', "Backend:[cpu, optix]", false, "cpu",
                       cmdline::oneof<std::string>("cpu", "optix"));
    a.add("gui", '\0', "Show GUI");
    a.parse_check(argc, argv);

    try {
        filesystem::path path(a.get<std::string>("scene"));
        std::string sceneName = a.get<std::string>("scene");

        if (path.extension() == "xml") {
            /* Add the parent directory of the scene file to the
               file resolver. That way, the XML file can reference
               resources (OBJ files, textures) using relative paths */
            getFileResolver()->prepend(path.parent_path());

            std::unique_ptr<Object> root(loadFromXML(sceneName));

            std::string backend = a.get<std::string>("backend");
            bool gui = a.exist("gui");
            if (backend == "cpu") {
                int thread = a.get<int>("thread");
                CPURenderer renderer;

                /* When the XML root object is a scene, start rendering it .. */
                if (root->getClassType() == Object::EScene) {
                    renderer.render(static_cast<Scene*>(root.get()), sceneName,
                                    gui, thread);
                }
            }
            else if (backend == "optix") {
                optix::OptixRenderer renderer(static_cast<Scene*>(root.get()));
                std::string filename = sceneName;
                size_t lastdot = filename.find_last_of(".");
                if (lastdot != std::string::npos)
                    filename.erase(lastdot, std::string::npos);
                renderer.render(filename, gui);
            }
            else {
                spdlog::critical("Fatal error: unknown backend:  {}", backend);
            }
        }
        else {
            spdlog::critical("Fatal error: unknown file: \"{}\", expected an "
                             "extension of type .xml",
                             path.str());
        }
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return -1;
    }
}