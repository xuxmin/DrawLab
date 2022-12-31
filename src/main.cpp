#include "core/base/common.h"
#include "core/bitmap/bitmap.h"
#include "core/math/math.h"
#include "core/math/matrix.h"
#include "core/parser/parser.h"
#include "editor/gui.h"
#include "tracer/scene.h"
#include "utils/pb.h"
#include <thread>

using namespace drawlab;

static void render(Scene* scene, const std::string& filename) {
    const Camera* camera = scene->getCamera();
    Vector2i outputSize = camera->getOutputSize();
    scene->getIntegrator()->preprocess(scene);

    int height = outputSize[1], width = outputSize[0];
    ImageBlock block(outputSize, camera->getReconstructionFilter());

    GUI gui(block);
    gui.init();

    std::thread render_thread([&] {
        /* Create a clone of the sampler for the current thread */
        std::unique_ptr<Sampler> sampler(scene->getSampler()->clone());
        const Integrator* integrator = scene->getIntegrator();

        /* For each pixel and pixel sample sample */
        ProgressBar bar;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (uint32_t i = 0; i < sampler->getSampleCount(); i++) {
                    Point2f pixelSample =
                        Point2f((float)w, (float)h) + sampler->next2D();
                    Point2f apertureSample = sampler->next2D();

                    // Sample a ray from the camera
                    Ray3f ray;
                    Color3f value =
                        camera->sampleRay(ray, pixelSample, apertureSample);

                    /* Compute the incident radiance */
                    value *= integrator->Li(scene, sampler.get(), ray);

                    block.put(pixelSample, value);
                }
            }
            bar.update(h / (float)(height));
        }
    });

    gui.start();
    render_thread.join();

    std::unique_ptr<Bitmap> bitmap(block.toBitmap());
    std::string outputName = filename;
    size_t lastdot = outputName.find_last_of(".");
    if (lastdot != std::string::npos)
        outputName.erase(lastdot, std::string::npos);
    bitmap->saveEXR(outputName);
    bitmap->savePNG(outputName);
}

int main(int argc, char** argv) {
    std::string sceneName = "";
    std::string exrName = "";
    filesystem::path path(argv[1]);

    try {
        if (path.extension() == "xml") {
            sceneName = argv[1];

            /* Add the parent directory of the scene file to the
               file resolver. That way, the XML file can reference
               resources (OBJ files, textures) using relative paths */
            getFileResolver()->prepend(path.parent_path());
        } else if (path.extension() == "exr") {
            exrName = argv[1];
        } else {
            cerr << "Fatal error: unknown file \"" << argv[1]
                 << "\", expected an extension of type .xml" << endl;
        }
    } catch (const std::exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return -1;
    }

    try {
        if (exrName != "") {
            Bitmap bitmap(exrName);
            ImageBlock block(bitmap);
            GUI gui(block);
            gui.init();
            gui.start();
        } else if (sceneName != "") {
            std::unique_ptr<Object> root(loadFromXML(sceneName));
            /* When the XML root object is a scene, start rendering it .. */
            if (root->getClassType() == Object::EScene)
                render(static_cast<Scene*>(root.get()), sceneName);
        }
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
}