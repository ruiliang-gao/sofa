# CMake generated Testfile for 
# Source directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/projects/runSofa/runSofa_test
# Build directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/applications/projects/runSofa/runSofa_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(runSofa_test "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/bin/Release/runSofa_test.exe")
  set_tests_properties(runSofa_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/projects/runSofa/runSofa_test/CMakeLists.txt;12;add_test;C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/projects/runSofa/runSofa_test/CMakeLists.txt;0;")
else()
  add_test(runSofa_test NOT_AVAILABLE)
endif()
