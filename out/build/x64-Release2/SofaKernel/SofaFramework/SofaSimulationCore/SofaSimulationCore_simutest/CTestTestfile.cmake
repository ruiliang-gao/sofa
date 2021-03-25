# CMake generated Testfile for 
# Source directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationCore/SofaSimulationCore_simutest
# Build directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaKernel/SofaFramework/SofaSimulationCore/SofaSimulationCore_simutest
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(SofaSimulationCore_simutest "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/bin/Release/SofaSimulationCore_simutest.exe")
  set_tests_properties(SofaSimulationCore_simutest PROPERTIES  _BACKTRACE_TRIPLES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationCore/SofaSimulationCore_simutest/CMakeLists.txt;15;add_test;C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationCore/SofaSimulationCore_simutest/CMakeLists.txt;0;")
else()
  add_test(SofaSimulationCore_simutest NOT_AVAILABLE)
endif()
