/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/testing/NumericTest.h>
#include <sofa/helper/logging/Messaging.h>

using namespace sofa::testing ;


using namespace sofa;
using namespace sofa::type;
using namespace sofa::helper;
using namespace sofa::defaulttype;

void test_transformInverse(Matrix4 const& M)
{
    Matrix4 M_inv;
    M_inv.transformInvert(M);
    Matrix4 res = M*M_inv;
    Matrix4 I;I.identity();
    EXPECT_MAT_NEAR(I, res, (SReal)1e-12);
}

TEST(MatTypesTest, transformInverse)
{
    test_transformInverse(Matrix4::s_identity);
    test_transformInverse(Matrix4::transformTranslation(Vector3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformScale(Vector3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformRotation(Quat<SReal>::fromEuler(M_PI_4,M_PI_2,M_PI/3.)));
}

TEST(MatTypesTest, setsub_vec)
{
    Matrix3 M = Matrix3::s_identity;
    Vector2 v(1.,2.);
    M.setsub(1,2,v);
    double exp[9]={1.,0.,0.,
                   0.,1.,1.,
                   0.,0.,2.};
    Matrix3 M_exp(exp);
    EXPECT_MAT_DOUBLE_EQ(M_exp, M);
}

TEST(MatTypesTest, isTransform)
{
    Matrix4 M;
    EXPECT_FALSE(M.isTransform());
    M.identity();
    EXPECT_TRUE(M.isTransform());
}

TEST(MatTypesTest, transpose)
{
    Matrix4 M(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12),
              Matrix4::Line(4, 14, 15, 1));

    Matrix4 Mnew;
    Mnew.transpose(M);

    Matrix4 Mtest(Matrix4::Line(16, 5, 9, 4), Matrix4::Line(2, 11, 7, 14), Matrix4::Line(3, 10, 6, 15),
                  Matrix4::Line(13, 8, 12, 1));

    EXPECT_EQ(Mnew, Mtest);
    EXPECT_EQ(M.transposed(), Mtest);

    M.transpose(M);
    EXPECT_EQ(M, Mtest);

    M = Matrix4(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12),
              Matrix4::Line(4, 14, 15, 1));

    M.transpose();
    EXPECT_EQ(M, Mtest);

    M.identity();
    EXPECT_EQ(M.transposed(), M);
}

TEST(MatTypesTest, nonSquareTranspose)
{
    Mat<3,4,double> M(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12));

    Mat<4,3,double> Mnew;
    Mnew.transpose(M);

    Mat<4,3,double> Mtest(Matrix3::Line(16,5,9), Matrix3::Line(2,11,7), Matrix3::Line(3,10,6), Matrix3::Line(13,8,12));

    EXPECT_EQ(Mnew, Mtest);
    EXPECT_EQ(M.transposed(), Mtest);
    EXPECT_EQ(M, Mtest.transposed());
}

TEST(MatTypesTest, invert22)
{
    Matrix2 M(Matrix2::Line(4.0, 7.0), Matrix2::Line(2.0, 6.0));
    Matrix2 Minv;
    const Matrix2 Mtest(Matrix2::Line(0.6,-0.7),
                        Matrix2::Line(-0.2,0.4));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}

TEST(MatTypesTest, invert33)
{
    Matrix3 M(Matrix3::Line(3., 0., 2.), Matrix3::Line(2., 0., -2.), Matrix3::Line(0., 1., 1.));
    Matrix3 Minv;
    const Matrix3 Mtest(Matrix3::Line(0.2, 0.2, 0.),
                        Matrix3::Line(-0.2, 0.3, 1.),
                        Matrix3::Line(0.2, -0.3, 0.));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}

TEST(MatTypesTest, invert55)
{
    Mat<5, 5, SReal> M(Mat<5, 5, SReal>::Line(-2.,  7.,  0.,  6., -2.),
                       Mat<5, 5, SReal>::Line( 1., -1.,  3.,  2.,  2.),
                       Mat<5, 5, SReal>::Line( 3.,  4.,  0.,  5.,  3.),
                       Mat<5, 5, SReal>::Line( 2.,  5., -4., -2.,  2.),
                       Mat<5, 5, SReal>::Line( 0.,  3., -1.,  1., -4.));
    Mat<5, 5, SReal> Minv;

    const Mat<5, 5, SReal> Mtest(Mat<5, 5, SReal>::Line(-289./1440., 11./90., 13./90., 31./1440., 101./360.),
                                 Mat<5, 5, SReal>::Line(37./360., 14./45., -8./45., 77./360., 7./90.),
                                 Mat<5, 5, SReal>::Line(17./288., 11./18., -5./18., 49./288., 11./72.),
                                 Mat<5, 5, SReal>::Line(1./1440., -29./90.,23./90.,-319./1440.,-29./360.),
                                 Mat<5, 5, SReal>::Line(1./16., 0., 0., 1./16., -1./4.));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}