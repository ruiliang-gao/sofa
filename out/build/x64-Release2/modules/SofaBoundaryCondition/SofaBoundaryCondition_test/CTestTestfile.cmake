# CMake generated Testfile for 
# Source directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaBoundaryCondition/SofaBoundaryCondition_test
# Build directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/modules/SofaBoundaryCondition/SofaBoundaryCondition_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(SofaBoundaryCondition_test "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/bin/Release/SofaBoundaryCondition_test.exe")
  set_tests_properties(SofaBoundaryCondition_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaBoundaryCondition/SofaBoundaryCondition_test/CMakeLists.txt;37;add_test;C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaBoundaryCondition/SofaBoundaryCondition_test/CMakeLists.txt;0;")
else()
  add_test(SofaBoundaryCondition_test NOT_AVAILABLE)
endif()
