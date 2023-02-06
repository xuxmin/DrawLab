#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"

namespace optix {

class DeviceContext;

class LaunchParam {
public:
    LaunchParam(const DeviceContext& device_context);

    void setupColorBuffer(int width, int height);

    void setupLights(const std::vector<Light>& lights);

    void setupCamera(const Camera& camera);

    void updateParamsBuffer();

    void getColorData(float3* pixels) const;

    const int getWidth() const { return m_width; }
    const int getHeight() const { return m_height; }
    const CUDABuffer& getParamsBuffer() const { return m_params_buffer; }

private:
    const DeviceContext& m_device_context;

    Params m_params;
    CUDABuffer m_params_buffer;

    int m_width;
    int m_height;
    CUDABuffer m_color_buffer;

    int m_light_num;
    CUDABuffer m_light_buffer;

    Camera m_camera;
};

}  // namespace optix