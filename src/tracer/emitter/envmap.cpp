#include "tracer/emitter.h"
#include "tracer/texture.h"
#include "core/bitmap/bitmap.h"
#include "core/utils/timer.h"
#include "optix/light/envmap.h"
#include "optix/host/cuda_buffer.h"
#include <memory>

namespace drawlab {

class EnvironmentMap final : public Emitter {
public:
    EnvironmentMap(const PropertyList& propList) {
        m_texture = static_cast<Texture2D*>(
            ObjectFactory::createInstance("bitmap", propList));
        m_size = m_texture->getResolution();
        m_visual = propList.getBoolean("visual", true);
        m_scale = propList.getFloat("scale", 1.f);
    }

    void sampleDirection(const Point2f& sample, const Intersection& its,
                         DirectionSample& ds,
                         Color3f& spectrum) const override {
        throw Exception("EnvironmentMap::sampleDirection() is not implemented!");
    }

    float pdfDirection(const DirectionSample& ds) const override {
        throw Exception("EnvironmentMap::pdfDirection() is not implemented!");
    }

    Color3f eval(const Intersection& its, const Vector3f wi) const {
        throw Exception("EnvironmentMap::eval() is not implemented!");
    }

    std::string toString() const {
        return tfm::format("EnvironmentMap[%s]", m_texture->toString());
    }

    bool isEnvironmentEmitter() const { return true; }

    void getOptixLight(optix::Light& light) const {
        std::shared_ptr<Bitmap> bitmap = m_texture->getBitmap();
        light.type = optix::Light::Type::ENVMAP;

        // TODO: release texture!!
        const optix::CUDATexture* tex = m_texture->createCUDATexture();
        light.envmap.env_tex = tex->getObject();
        light.envmap.env_size = make_int2(m_size[0], m_size[1]);

        // TODO: release buffer!!
        optix::CUDABuffer buffer;
        buffer.allocAndUpload(m_env_accel);
        light.envmap.env_accel = (optix::EnvAccel*)buffer.devicePtr();

        light.envmap.visual = m_visual;
        light.envmap.scale = m_scale;
    }

    void activate() {
        /**
         * ------------------------------------------------------------
         * Environment Important Sampling
         * ------------------------------------------------------------
         *
         * We want to sample a light direction based on the envmap, only
         * have two steps:
         *  1. important sample an envmap pixel
         *  2. uniformly sample spherical area of the pixel.
         *
         * Some key points:
         *  1. each envmap pixel corresponds to a spherical area, and their
         *     area are different!
         *  2. In order to important sample an envmap pixel in O(1), we
         *     should build an alias map first.
         */
        std::shared_ptr<Bitmap> bitmap = m_texture->getBitmap();
        unsigned int rx = bitmap->getWidth();
        unsigned int ry = bitmap->getHeight();
        float* pixels = bitmap->getPtr();
        m_env_accel.resize(rx * ry);

        // Calculate the importance of each pixel(consider pixel area 
        // and pixel intensity)
        float* importance_data = (float*)malloc(rx * ry * sizeof(float));
        float cos_theta0 = 1.0f;
        const float step_phi = (float)(2.0 * M_PIf) / (float)rx;
        const float step_theta = (float)M_PIf / (float)ry;
	    for (unsigned int y = 0; y < ry; ++y) {
            const float theta1 = (float)(y + 1) * step_theta;
            const float cos_theta1 = cos(theta1);
            const float area = (cos_theta0 - cos_theta1) * step_phi;
            cos_theta0 = cos_theta1;

            for (unsigned int x = 0; x < rx; ++x) {
                const unsigned int idx = y * rx + x;
                const unsigned int idx3 = idx * 3;
                importance_data[idx] = area * std::max(pixels[idx3], std::max(pixels[idx3 + 1], pixels[idx3 + 2]));
            }
        }

        // Build alias map
	    const float inv_env_integral = 1.0f / build_alias_map(importance_data, rx * ry, m_env_accel);
	    free(importance_data);

        // Calculate pdf
        for (unsigned int i = 0; i < rx * ry; ++i) {
            const unsigned int idx3 = i * 3;
            m_env_accel[i].pdf = std::max(pixels[idx3], std::max(pixels[idx3 + 1], pixels[idx3 + 2])) * inv_env_integral;
        }
    }

private:
    static float build_alias_map(const float* data, const unsigned int size,
                                 std::vector<optix::EnvAccel>& accel) {
        // create qs (normalized)
        float sum = 0.0f;
        for (unsigned int i = 0; i < size; ++i)
            sum += data[i];

        for (unsigned int i = 0; i < size; ++i)
            accel[i].q = (float)((float)size * data[i] / sum);

        // create partition table
        unsigned int* partition_table =
            (unsigned int*)malloc(size * sizeof(unsigned int));
        unsigned int s = 0u, large = size;
        for (unsigned int i = 0; i < size; ++i)
            partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] =
                accel[i].alias = i;

        // create alias map
        for (s = 0; s < large && large < size; ++s) {
            const unsigned int j = partition_table[s],
                               k = partition_table[large];
            accel[j].alias = k;
            accel[k].q += accel[j].q - 1.0f;
            large = (accel[k].q < 1.0f) ? (large + 1u) : large;
        }
        free(partition_table);
        return sum;
    }

private:
    Texture2D* m_texture;
    Vector3i m_size;
    bool m_visual;
    float m_scale;
    std::vector<optix::EnvAccel> m_env_accel;
};

REGISTER_CLASS(EnvironmentMap, "envmap");

}  // namespace drawlab