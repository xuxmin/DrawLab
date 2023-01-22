#include "core/base/common.h"
#include "core/bitmap/bitmap.h"
#include "core/math/math.h"
#include "core/math/matrix.h"
#include "core/parser/parser.h"
#include "core/bitmap/block.h"
#include "editor/gui.h"
#include "tracer/scene.h"
#include "core/utils/pb.h"
#include "core/utils/timer.h"
#include <thread>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace drawlab;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    #define SHOW_GUI
#else
#endif

static int threadCount = 4;


static void renderBlock(const Scene* scene, Sampler* sampler,
                        ImageBlock& block) {
    const Camera* camera = scene->getCamera();
    const Integrator* integrator = scene->getIntegrator();

    Point2i offset = block.getOffset();
    Vector2i size  = block.getSize();

    /* Clear the block contents */
    block.clear();

    /* For each pixel and pixel sample sample */
    for (int y = 0; y < size.y(); ++y) {
        for (int x = 0; x < size.x(); ++x) {
            for (uint32_t i = 0; i < sampler->getSampleCount(); ++i) {
                Point2f pixelSample = Point2f((float) (x + offset.x()), (float) (y + offset.y())) + sampler->next2D();
                Point2f apertureSample = sampler->next2D();

                /* Sample a ray from the camera */
                Ray3f ray;
                Color3f value = camera->sampleRay(ray, pixelSample, apertureSample);

                /* Compute the incident radiance */
                value *= integrator->Li(scene, sampler, ray);

                /* Store in the image block */
                block.put(pixelSample, value);
            }
        }
    }
}

static void render(Scene* scene, const std::string& filename) {
    const Camera* camera = scene->getCamera();
    Vector2i outputSize = camera->getOutputSize();
    scene->getIntegrator()->preprocess(scene);

    const int BLOCK_SIZE = 32;

    /* Create a block generator (i.e. a work scheduler) */
    BlockGenerator blockGenerator(outputSize, BLOCK_SIZE);

    int height = outputSize[1], width = outputSize[0];
    ImageBlock result(outputSize, camera->getReconstructionFilter());

#ifdef SHOW_GUI
    GUI gui(result);
    gui.init();
#endif

    std::thread render_thread([&] {
        tbb::task_scheduler_init init(threadCount);

        tbb::blocked_range<int> range(0, blockGenerator.getBlockCount());

        ProgressBar bar;
        int cnt = 0;
        int total_cnt = blockGenerator.getBlockCount();
        Timer timer;

        auto map = [&](const tbb::blocked_range<int> &range) {
            /* Allocate memory for a small image block to be rendered
               by the current thread */
            ImageBlock block(Vector2i(BLOCK_SIZE), camera->getReconstructionFilter());

            /* Create a clone of the sampler for the current thread */
            std::unique_ptr<Sampler> sampler(scene->getSampler()->clone());

            for (int i = range.begin(); i < range.end(); ++i) {
                /* Request an image block from the block generator */
                blockGenerator.next(block);

                /* Inform the sampler about the block to be rendered */
                sampler->prepare(block);

                /* Render all contained pixels */
                renderBlock(scene, sampler.get(), block);

                /* The image block has been processed. Now add it to
                   the "big" block that represents the entire image */
                result.put(block);

                bar.update(cnt++ / (float)(total_cnt));
            }
        };

        /// Default: parallel rendering
        tbb::parallel_for(range, map);

        /// (equivalent to the following single-threaded call)
        // map(range);
        cout << "done. (took " << timer.elapsedString() << ")" << endl;
    });

#ifdef SHOW_GUI
    gui.start();
#endif

    render_thread.join();

    std::unique_ptr<Bitmap> bitmap(result.toBitmap());
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