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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa::simpleapi ;

#include "Node_test.h"

namespace sofa
{

TEST( Node_test, getPathName)
{
    /* create trivial DAG :
     *
     * A
     * |\
     * B C
     * |
     * D
     *
     */
    EXPECT_MSG_NOEMIT(Error, Warning);

    Node::SPtr root = sofa::simpleapi::createNode("A");
    Node::SPtr B = createChild(root, "B");
    Node::SPtr D = createChild(B, "D");
    BaseObject::SPtr C = core::objectmodel::New<Dummy>("C");
    root->addObject(C);

    EXPECT_STREQ(root->getPathName().c_str(), "/");
    EXPECT_STREQ(B->getPathName().c_str(), "/B");
    EXPECT_STREQ(C->getPathName().c_str(), "/C");
    EXPECT_STREQ(D->getPathName().c_str(), "/B/D");
}

TEST(Node_test, addObject)
{
    sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    BaseObject::SPtr A = core::objectmodel::New<Dummy>("A");
    BaseObject::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);

    // adds a second object after the last one.
    root->addObject(B);
    auto b = dynamic_cast< sofa::core::objectmodel::BaseContext*>(root.get());
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->getObjects()[0]->getName(), "A");
    ASSERT_EQ(b->getObjects()[1]->getName(), "B");
}

TEST(Node_test, addObjectAtFront)
{
    sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    BaseObject::SPtr A = core::objectmodel::New<Dummy>("A");
    BaseObject::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);

    // adds a second object before the first one.
    root->addObject(B, sofa::core::objectmodel::TypeOfInsertion::AtBegin);
    auto b = dynamic_cast< sofa::core::objectmodel::BaseContext*>(root.get());
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->getObjects()[0]->getName(), "B");
    ASSERT_EQ(b->getObjects()[1]->getName(), "A");
}


}// namespace sofa







