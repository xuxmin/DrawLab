#include "core/math/normal.h"
#include "core/math/vector.h"
#include <gtest/gtest.h>

using namespace drawlab;

TEST(NormalTest, Construction) {
    TNormal3<float> n0;
    TNormal3<float> n1(1, 2, 3);
    TNormal3<float> n2(n0);
    TNormal3<float> n3 = n0;
}

TEST(NormalTest, Operation) {
    TNormal3<float> n0;
    n0[0] = 1;
    n0[2] = 10;

    EXPECT_EQ(n0[0], n0.ptr()[0]);
    EXPECT_EQ(n0.coeff(0), 1);
    EXPECT_EQ(n0.coeff(1), 0);

    TNormal3<float> v0(3, 4, 5);

    EXPECT_EQ(v0.squaredLength(), 50);
    EXPECT_FLOAT_EQ(v0.length(), (float)sqrt(50));

    TNormal3<float> v1 = v0.normalized();
    EXPECT_FLOAT_EQ(v1[0], 3 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v1[1], 4 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v1[2], 5 / (float)sqrt(50));
    v0.normalize();
    EXPECT_FLOAT_EQ(v0[0], 3 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v0[1], 4 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v0[2], 5 / (float)sqrt(50));

    TVector<3, float> vector = {3, 4, 5};
    TNormal3<float> normal = {3, 4, 5};
    EXPECT_FLOAT_EQ(normal.dot(vector), 50);
}
