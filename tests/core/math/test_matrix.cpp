#include "core/math/matrix.h"
#include <gtest/gtest.h>

using namespace drawlab;

typedef TVector<4, float> Vector4f;

TEST(MatrixTest, Operation) {

    Matrix4f m;

    m.setRow(0, Vector4f(1, 2, 3, 4));
    m.setRow(1, Vector4f(-1, -2, -3, -4));
    m.setRow(2, Vector4f(9, 8, 7, 6));
    m.setRow(3, Vector4f(-1, 4, 6, 1));

    Matrix4f copy_m = m;

    Vector4f v0 = {1, -1, 9, -1};
    EXPECT_EQ(copy_m.col(0), v0);
    Vector4f v1 = {9, 8, 7, 6};
    EXPECT_EQ(copy_m.row(2), v1);

    copy_m.setCol(0, v1);
    Vector4f v2 = {7, 8, 7, 6};
    EXPECT_EQ(copy_m.row(2), v2);
}