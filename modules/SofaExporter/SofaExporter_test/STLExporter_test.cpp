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
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <SofaSimulationGraph/SimpleApi.h>

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::execparams::defaultInstance; 

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <SofaBaseUtils/initSofaBaseUtils.h>

#include <filesystem>
namespace {
std::string tempdir = std::filesystem::temp_directory_path().string() ;


class STLExporter_test : public BaseSimulationTest {
public:
    /// remove the file created...
    std::vector<std::string> dataPath ;

    void SetUp() override
    {
        sofa::component::initSofaBaseUtils(); // needed to instanciate RequiredPlugin
        sofa::simpleapi::importPlugin("SofaOpenglVisual");
    }

    void TearDown() override
    {
        return ;
        for(auto& pathToRemove : dataPath)
        {
            if(FileSystem::exists(pathToRemove))
               FileSystem::removeAll(pathToRemove) ;
       }
    }

    void checkBasicBehavior(const std::string& filename, std::vector<std::string> pathes){
        dataPath = pathes ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <RequiredPlugin name='SofaLoader' />                           \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <MeshOBJLoader name='loader' filename='mesh/liver-smooth.obj'/> \n"
                "   <OglModel src='@loader'/>                                      \n"
                "   <STLExporter name='exporter1' printLog='true' filename='"<< filename << "' exportAtBegin='true' /> \n"
                "</Node>                                                           \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene1.str().c_str(),
                                                          scene1.str().size()) ;

        ASSERT_NE(root.get(), nullptr) << scene1.str() ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        sofa::simulation::getSimulation()->animate(root.get(), 0.5);

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'"<< std::endl
                                                           << "================= scene dump ==========================="
                                                           << scene1.str() ;
        }
    }


    void checkSimulationWriteEachNbStep(const std::string& filename, std::vector<std::string> pathes, unsigned int numstep){
        dataPath = pathes ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <MeshOBJLoader name='loader' filename='mesh/liver-smooth.obj'/> \n"
                "   <OglModel src='@loader'/>                                      \n"
                "   <STLExporter name='exporterA' printLog='true' filename='"<< filename << "' exportEveryNumberOfSteps='5' /> \n"
                "</Node>                                                           \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene1.str().c_str(),
                                                          scene1.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        for(unsigned int i=0;i<numstep;i++)
        {
            sofa::simulation::getSimulation()->animate(root.get(), 0.5);
        }

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'";
        }
    }
};

/// run the tests
TEST_F( STLExporter_test, checkBasicBehavior) {
    ASSERT_NO_THROW( this->checkBasicBehavior("outfile", {"outfile.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorNoFileName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("", {"exporter1.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/outfile", {tempdir+"/outfile.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid/outfile", {tempdir+"/invalid"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidLongSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid1/invalid2/invalid3/outfile", {tempdir+"/invalid1/invalid2/invalid3"})) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidRelativeDirName) {
   ASSERT_NO_THROW( this->checkBasicBehavior("./invalidPath/outfile", {"./invalidPath"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInValidDir) {
   this->checkBasicBehavior(tempdir, {tempdir+"/exporter1.stl"})  ;
}

TEST_F( STLExporter_test, checkSimulationWriteEachNbStep) {
   this->checkSimulationWriteEachNbStep(tempdir, {tempdir+"/exporterA00001.stl",
                                                 tempdir+"/exporterA00002.stl",
                                                 tempdir+"/exporterA00003.stl",
                                                 tempdir+"/exporterA00004.stl"}, 20)  ;
}
}
