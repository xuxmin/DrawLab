#include <iostream>
#include <tbb/mutex.h>

namespace drawlab {

class ProgressBar {
public:
    ProgressBar(const float barWidth = 70.f) : m_barWidth(barWidth) {}
    void update(float progress) {
        tbb::mutex::scoped_lock lock(m_mutex);
        std::cout << "[";
        int pos = m_barWidth * progress;
        for (int i = 0; i < m_barWidth; ++i) {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        if (progress >= 1.0) {
            std::cout << std::endl;
        }
    }

private:
    float m_barWidth;
    tbb::mutex m_mutex;
};

}  // namespace drawlab
