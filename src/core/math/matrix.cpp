#include "core/math/matrix.h"
#include <iostream>


namespace drawlab {

/// Get the inverse matrix
Matrix4f Matrix4f::inv() const {
    float mm[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mm[i * 4 + j] = m[i][j];
        }
    }

    float inv[16], det;
    inv[0] = mm[5] * mm[10] * mm[15] - mm[5] * mm[11] * mm[14] -
             mm[9] * mm[6] * mm[15] + mm[9] * mm[7] * mm[14] +
             mm[13] * mm[6] * mm[11] - mm[13] * mm[7] * mm[10];

    inv[4] = -mm[4] * mm[10] * mm[15] + mm[4] * mm[11] * mm[14] +
             mm[8] * mm[6] * mm[15] - mm[8] * mm[7] * mm[14] -
             mm[12] * mm[6] * mm[11] + mm[12] * mm[7] * mm[10];

    inv[8] = mm[4] * mm[9] * mm[15] - mm[4] * mm[11] * mm[13] -
             mm[8] * mm[5] * mm[15] + mm[8] * mm[7] * mm[13] +
             mm[12] * mm[5] * mm[11] - mm[12] * mm[7] * mm[9];

    inv[12] = -mm[4] * mm[9] * mm[14] + mm[4] * mm[10] * mm[13] +
              mm[8] * mm[5] * mm[14] - mm[8] * mm[6] * mm[13] -
              mm[12] * mm[5] * mm[10] + mm[12] * mm[6] * mm[9];

    inv[1] = -mm[1] * mm[10] * mm[15] + mm[1] * mm[11] * mm[14] +
             mm[9] * mm[2] * mm[15] - mm[9] * mm[3] * mm[14] -
             mm[13] * mm[2] * mm[11] + mm[13] * mm[3] * mm[10];

    inv[5] = mm[0] * mm[10] * mm[15] - mm[0] * mm[11] * mm[14] -
             mm[8] * mm[2] * mm[15] + mm[8] * mm[3] * mm[14] +
             mm[12] * mm[2] * mm[11] - mm[12] * mm[3] * mm[10];

    inv[9] = -mm[0] * mm[9] * mm[15] + mm[0] * mm[11] * mm[13] +
             mm[8] * mm[1] * mm[15] - mm[8] * mm[3] * mm[13] -
             mm[12] * mm[1] * mm[11] + mm[12] * mm[3] * mm[9];

    inv[13] = mm[0] * mm[9] * mm[14] - mm[0] * mm[10] * mm[13] -
              mm[8] * mm[1] * mm[14] + mm[8] * mm[2] * mm[13] +
              mm[12] * mm[1] * mm[10] - mm[12] * mm[2] * mm[9];

    inv[2] = mm[1] * mm[6] * mm[15] - mm[1] * mm[7] * mm[14] -
             mm[5] * mm[2] * mm[15] + mm[5] * mm[3] * mm[14] +
             mm[13] * mm[2] * mm[7] - mm[13] * mm[3] * mm[6];

    inv[6] = -mm[0] * mm[6] * mm[15] + mm[0] * mm[7] * mm[14] +
             mm[4] * mm[2] * mm[15] - mm[4] * mm[3] * mm[14] -
             mm[12] * mm[2] * mm[7] + mm[12] * mm[3] * mm[6];

    inv[10] = mm[0] * mm[5] * mm[15] - mm[0] * mm[7] * mm[13] -
              mm[4] * mm[1] * mm[15] + mm[4] * mm[3] * mm[13] +
              mm[12] * mm[1] * mm[7] - mm[12] * mm[3] * mm[5];

    inv[14] = -mm[0] * mm[5] * mm[14] + mm[0] * mm[6] * mm[13] +
              mm[4] * mm[1] * mm[14] - mm[4] * mm[2] * mm[13] -
              mm[12] * mm[1] * mm[6] + mm[12] * mm[2] * mm[5];

    inv[3] = -mm[1] * mm[6] * mm[11] + mm[1] * mm[7] * mm[10] +
             mm[5] * mm[2] * mm[11] - mm[5] * mm[3] * mm[10] -
             mm[9] * mm[2] * mm[7] + mm[9] * mm[3] * mm[6];

    inv[7] = mm[0] * mm[6] * mm[11] - mm[0] * mm[7] * mm[10] -
             mm[4] * mm[2] * mm[11] + mm[4] * mm[3] * mm[10] +
             mm[8] * mm[2] * mm[7] - mm[8] * mm[3] * mm[6];

    inv[11] = -mm[0] * mm[5] * mm[11] + mm[0] * mm[7] * mm[9] +
              mm[4] * mm[1] * mm[11] - mm[4] * mm[3] * mm[9] -
              mm[8] * mm[1] * mm[7] + mm[8] * mm[3] * mm[5];

    inv[15] = mm[0] * mm[5] * mm[10] - mm[0] * mm[6] * mm[9] -
              mm[4] * mm[1] * mm[10] + mm[4] * mm[2] * mm[9] +
              mm[8] * mm[1] * mm[6] - mm[8] * mm[2] * mm[5];

    det = mm[0] * inv[0] + mm[1] * inv[4] + mm[2] * inv[8] + mm[3] * inv[12];

    if (det == 0) {
        throw std::runtime_error("Can't compute inverse matrix.");
    }

    det = (float)1.0 / det;
    Matrix4f ret = {inv[0],  inv[1],  inv[2],  inv[3], inv[4],  inv[5],
                    inv[6],  inv[7],  inv[8],  inv[9], inv[10], inv[11],
                    inv[12], inv[13], inv[14], inv[15]};

    ret = ret * det;

    return ret;
}

}  // namespace drawlab