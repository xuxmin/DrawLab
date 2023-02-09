#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"

namespace optix {

class DeviceContext;

class LaunchParam {
public:
    LaunchParam(const DeviceContext& device_context);

    ~LaunchParam();

    void setupColorBuffer(int width, int height);

    void setupLights(const std::vector<Light>& lights);

    void setupCamera(const Camera& camera);

    void setupSampler(const int spp);

    void updateParamsBuffer();

    void getColorData(float3* pixels) const;

    const int getWidth() const { return m_width; }
    const int getHeight() const { return m_height; }
    const CUDABuffer& getParamsBuffer() const { return m_params_buffer; }

    void resetFrameIndex();
    void accFrameIndex();

private:
    const DeviceContext& m_device_context;

    Params m_params;
    CUDABuffer m_params_buffer;

    int m_width;
    int m_height;
    CUDABuffer m_color_buffer;

    int m_frame_index;

    int m_light_num;
    CUDABuffer m_light_buffer;

    Camera m_camera;

    bool dirty;
};

}  // namespace optix