#include "core/base/common.h"
#include "core/parser/parser.h"
#include "tracer/backend/cpu/cpu_renderer.h"
#include "tracer/scene.h"
#include "cmdline.h"
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
            if (backend == "cpu") {
                int thread = a.get<int>("thread");
                bool gui = a.exist("gui");
                CPURenderer renderer;

                /* When the XML root object is a scene, start rendering it .. */
                if (root->getClassType() == Object::EScene) {
                    renderer.render(static_cast<Scene*>(root.get()), sceneName,
                                    gui, thread);
                }
            } else if (backend == "optix") {
            } else {
                cerr << "Fatal error: unknown backend: " << backend << endl;
            }
        } else {
            cerr << "Fatal error: unknown file \"" << path.str()
                 << "\", expected an extension of type .xml" << endl;
        }
    } catch (const std::exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return -1;
    }
}