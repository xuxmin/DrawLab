#include "core/math/point.h"
#include "core/math/vector.h"
#include <gtest/gtest.h>

using namespace drawlab;

TEST(PointTest, Construction) {
    TPoint<3, float> p0;
    TPoint<3, float> p1(0.f);
    TPoint<3, float> p2(0.f, 0.f, 0.f);
    TPoint<3, float> p3(p0);
    TPoint<3, float> p4({0.f, 0.f, 0.f});
}

TEST(PointTest, Operation) {
    TPoint<3, float> p0;
    p0[0] = 1;
    p0[2] = 10;

    EXPECT_EQ(p0[0], p0.ptr()[0]);
    EXPECT_EQ(p0.coeff(0), 1);
    EXPECT_EQ(p0.coeff(1), 0);
    EXPECT_EQ(p0.maxCoeff(), 10);

    TPoint<3, float> p2 = {1, 0, 10};
    EXPECT_EQ(p2 == p0, true);

    p0.setConstant(1);
    p2.setConstant(2);

    TPoint<3, float> p3 = {2, 2, 2};

    EXPECT_EQ(p0 * p2, p3);
    EXPECT_EQ(2 * p0, p3);
    EXPECT_EQ(p0 * 2, p3);
    EXPECT_EQ(p3 / 2, p0);
    EXPECT_EQ(p3 / p0, p3);
    EXPECT_EQ(p0 + p0, p3);

    TVector<3, float> v0 = {1, 1, 1};
    EXPECT_EQ(p3 - p0, v0);

    p0[0] = 1;
    p0[1] = 0;
    p0[2] = 10;

    p0 += 10;
    EXPECT_EQ(p0[0], 11);
    EXPECT_EQ(p0[1], 10);
    EXPECT_EQ(p0[2], 20);

    p0 -= 1;
    EXPECT_EQ(p0[0], 10);
    EXPECT_EQ(p0[1], 9);
    EXPECT_EQ(p0[2], 19);

    p0 *= 10;
    EXPECT_EQ(p0[0], 100);
    EXPECT_EQ(p0[1], 90);
    EXPECT_EQ(p0[2], 190);

    p0 /= 5;
    EXPECT_EQ(p0[0], 20);
    EXPECT_EQ(p0[1], 18);
    EXPECT_EQ(p0[2], 38);

    p0 += p0;
    EXPECT_EQ(p0[0], 40);
    EXPECT_EQ(p0[1], 36);
    EXPECT_EQ(p0[2], 76);

    p0 *= p0;
    EXPECT_EQ(p0[0], 40 * 40);
    EXPECT_EQ(p0[1], 36 * 36);
    EXPECT_EQ(p0[2], 76 * 76);

    p0 /= p0;
    EXPECT_EQ(p0[0], 1);
    EXPECT_EQ(p0[1], 1);
    EXPECT_EQ(p0[2], 1);

    p0 -= p0;
    EXPECT_EQ(p0[0], 0);
    EXPECT_EQ(p0[1], 0);
    EXPECT_EQ(p0[2], 0);
}
