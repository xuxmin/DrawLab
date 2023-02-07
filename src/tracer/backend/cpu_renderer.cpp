#include "tracer/backend/cpu_renderer.h"
#include "core/bitmap/bitmap.h"
#include "core/bitmap/block.h"
#include "core/utils/pb.h"
#include "core/utils/timer.h"
#include "editor/gui.h"
#include "tracer/scene.h"
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

namespace drawlab {

CPURenderer::CPURenderer(Scene* scene) : m_scene(scene) {
    const Camera* camera = m_scene->getCamera();
    Vector2i outputSize = camera->getOutputSize();
    m_block = new ImageBlock(outputSize, camera->getReconstructionFilter());
    m_display = nullptr;
}

void CPURenderer::renderBlock(const Scene* scene, Sampler* sampler,
                              ImageBlock& block) {
    const Camera* camera = scene->getCamera();
    const Integrator* integrator = scene->getIntegrator();

    Point2i offset = block.getOffset();
    Vector2i size = block.getSize();

    /* Clear the block contents */
    block.clear();

    /* For each pixel and pixel sample sample */
    for (int y = 0; y < size.y(); ++y) {
        for (int x = 0; x < size.x(); ++x) {
            for (uint32_t i = 0; i < sampler->getSampleCount(); ++i) {
                Point2f pixelSample =
                    Point2f((float)(x + offset.x()), (float)(y + offset.y())) +
                    sampler->next2D();
                Point2f apertureSample = sampler->next2D();

                /* Sample a ray from the camera */
                Ray3f ray;
                Color3f value =
                    camera->sampleRay(ray, pixelSample, apertureSample);

                /* Compute the incident radiance */
                value *= integrator->Li(scene, sampler, ray);

                /* Store in the image block */
                block.put(pixelSample, value);
            }
        }
    }
}

void CPURenderer::renderAsync(const std::string& filename, bool gui,
                              const int thread_count) {
    const Camera* camera = m_scene->getCamera();
    Vector2i outputSize = camera->getOutputSize();
    m_scene->getIntegrator()->preprocess(m_scene);

    if (m_scene->getEmitters().empty()) {
        throw Exception("There is not emitter in the scene.");
        exit(-1);
    }

    const int BLOCK_SIZE = 32;

    /* Create a block generator (i.e. a work scheduler) */
    BlockGenerator blockGenerator(outputSize, BLOCK_SIZE);

    std::thread render_thread([&] {
        tbb::task_scheduler_init init(thread_count);

        tbb::blocked_range<int> range(0, blockGenerator.getBlockCount());

        ProgressBar bar;
        int cnt = 0;
        int total_cnt = blockGenerator.getBlockCount();
        bar.update(0 / (float)(total_cnt));
        Timer timer;

        auto map = [&](const tbb::blocked_range<int>& range) {
            /* Allocate memory for a small image block to be rendered
               by the current thread */
            ImageBlock block(Vector2i(BLOCK_SIZE),
                             camera->getReconstructionFilter());

            /* Create a clone of the sampler for the current thread */
            std::unique_ptr<Sampler> sampler(m_scene->getSampler()->clone());

            for (int i = range.begin(); i < range.end(); ++i) {
                /* Request an image block from the block generator */
                blockGenerator.next(block);

                /* Inform the sampler about the block to be rendered */
                sampler->prepare(block);

                /* Render all contained pixels */
                renderBlock(m_scene, sampler.get(), block);

                /* The image block has been processed. Now add it to
                   the "big" block that represents the entire image */
                m_block->put(block);

                bar.update(cnt++ / (float)(total_cnt));
            }
        };

        /// Default: parallel rendering
        tbb::parallel_for(range, map);

        /// (equivalent to the following single-threaded call)
        // map(range);
        spdlog::info("Render done. (took {})", timer.elapsedString());
    });

    if (gui) {
        GUI gui(outputSize[0], outputSize[1]);
        gui.setRenderer(this);
        gui.init();
        gui.start();
    }

    render_thread.join();

    std::unique_ptr<Bitmap> bitmap(m_block->toBitmap());
    std::string outputName = filename;
    size_t lastdot = outputName.find_last_of(".");
    if (lastdot != std::string::npos)
        outputName.erase(lastdot, std::string::npos);
    bitmap->saveEXR(outputName);
    bitmap->savePNG(outputName);
}

void CPURenderer::init() {
    m_display = new opengl::Display(opengl::Display::BufferImageFormat::FLOAT3);
}

void CPURenderer::render() {
    const int w = GUI::window_width;
    const int h = GUI::window_height;

    m_block->lock();
    Bitmap* bitmap = m_block->toBitmap();
    bitmap->flipud();
    bitmap->resize(h, w);
    unsigned int pbo = m_display->getPBO(w, h, bitmap->getPtr());
    m_display->display(w, h, w, h, pbo);
    delete bitmap;
    m_block->unlock();
}

void CPURenderer::resize(size_t w, size_t h) {}

void CPURenderer::keyEvent(char key) {}

void CPURenderer::cursorEvent(float x, float y, unsigned char keys) {}

void CPURenderer::scrollEvent(float offset_x, float offset_y) {}

void CPURenderer::mouseButtonEvent(int button, int event) {}

}  // namespace drawlab