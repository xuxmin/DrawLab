#include "core/math/vector.h"
#include <gtest/gtest.h>

using namespace drawlab;

TEST(VectorTest, Construction) {
    TVector<3, float> v0;
    TVector<3, float> v1(0.f);
    TVector<3, float> v2(0.f, 0.f, 0.f);
    TVector<3, float> v3(v0);
    TVector<3, float> v4({0.f, 0.f, 0.f});
}

TEST(VectorTest, Operation) {
    TVector<3, float> v0;
    v0[0] = 1;
    v0[2] = 10;

    EXPECT_EQ(v0[0], v0.ptr()[0]);
    EXPECT_EQ(v0.coeff(0), 1);
    EXPECT_EQ(v0.coeff(1), 0);
    EXPECT_EQ(v0.maxCoeff(), 10);

    TVector<3, float> v1 = v0.cwiseInverse();
    EXPECT_EQ(v1[2], (float)1 / 10);

    TVector<3, float> v2 = {1, 0, 10};
    EXPECT_EQ(v2 == v0, true);
    EXPECT_EQ(v2 == v1, false);
    EXPECT_EQ(v2 != v1, true);

    EXPECT_EQ(v1 + v2, v1 - v0 + v2 + v0);

    TVector<3, float> v3(1, 0, 100);
    EXPECT_EQ(v0 * v2, v3);

    TVector<3, float> v4(10, 0, 100);
    EXPECT_EQ(v0 * 10, v4);
    EXPECT_EQ(10 * v0, v4);

    TVector<3, float> v5(10, 0, 100);
    EXPECT_EQ(v4 / 10, v0);

    v0 += 10;
    EXPECT_EQ(v0[0], 11);
    EXPECT_EQ(v0[1], 10);
    EXPECT_EQ(v0[2], 20);

    v0 -= 1;
    EXPECT_EQ(v0[0], 10);
    EXPECT_EQ(v0[1], 9);
    EXPECT_EQ(v0[2], 19);

    v0 *= 10;
    EXPECT_EQ(v0[0], 100);
    EXPECT_EQ(v0[1], 90);
    EXPECT_EQ(v0[2], 190);

    v0 /= 5;
    EXPECT_EQ(v0[0], 20);
    EXPECT_EQ(v0[1], 18);
    EXPECT_EQ(v0[2], 38);

    v0 += v0;
    EXPECT_EQ(v0[0], 40);
    EXPECT_EQ(v0[1], 36);
    EXPECT_EQ(v0[2], 76);

    v0 *= v0;
    EXPECT_EQ(v0[0], 40 * 40);
    EXPECT_EQ(v0[1], 36 * 36);
    EXPECT_EQ(v0[2], 76 * 76);

    v0 /= v0;
    EXPECT_EQ(v0[0], 1);
    EXPECT_EQ(v0[1], 1);
    EXPECT_EQ(v0[2], 1);

    v0 -= v0;
    EXPECT_EQ(v0[0], 0);
    EXPECT_EQ(v0[1], 0);
    EXPECT_EQ(v0[2], 0);
}

TEST(VectorTest, Function) {
    TVector<3, float> v0(3, 4, 5);

    EXPECT_EQ(v0.squaredLength(), 50);
    EXPECT_FLOAT_EQ(v0.length(), (float)sqrt(50));

    TVector<3, float> v1 = v0.normalized();
    EXPECT_FLOAT_EQ(v1[0], 3 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v1[1], 4 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v1[2], 5 / (float)sqrt(50));
    v0.normalize();
    EXPECT_FLOAT_EQ(v0[0], 3 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v0[1], 4 / (float)sqrt(50));
    EXPECT_FLOAT_EQ(v0[2], 5 / (float)sqrt(50));

    EXPECT_FLOAT_EQ(v0.prod(), (float)6 / (float)5 / (float)sqrt(50));

    TVector<2, float> v4(1, 2);
    TVector<2, float> v5(-1, 2);
    TVector<2, float> v4_dot_v5(-1, 4);
    EXPECT_EQ(v4.dot(v5), 3);
    EXPECT_EQ(v4.cross(v5), 4);
    EXPECT_EQ(v5.cross(v4), -4);

    TVector<3, float> v6(1, 2, 3);
    TVector<3, float> v7(10, -2, 20);
    TVector<3, float> v6_v7(46, 10, -22);
    TVector<3, float> v7_v6(-46, -10, 22);

    EXPECT_EQ(v6.cross(v7), v6_v7);
    EXPECT_EQ(v7.cross(v6), v7_v6);
}