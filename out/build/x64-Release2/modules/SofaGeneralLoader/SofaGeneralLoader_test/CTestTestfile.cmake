# CMake generated Testfile for 
# Source directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGeneralLoader/SofaGeneralLoader_test
# Build directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/modules/SofaGeneralLoader/SofaGeneralLoader_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(SofaGeneralLoader_test "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/bin/Release/SofaGeneralLoader_test.exe")
  set_tests_properties(SofaGeneralLoader_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGeneralLoader/SofaGeneralLoader_test/CMakeLists.txt;14;add_test;C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGeneralLoader/SofaGeneralLoader_test/CMakeLists.txt;0;")
else()
  add_test(SofaGeneralLoader_test NOT_AVAILABLE)
endif()
