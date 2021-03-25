# CMake generated Testfile for 
# Source directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaBaseUtils/SofaBaseUtils_test
# Build directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release/SofaKernel/SofaBase/SofaBaseVisual/SofaBaseUtils_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(SofaBaseUtils_test "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release/bin/Release/SofaBaseUtils_test.exe")
  set_tests_properties(SofaBaseUtils_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaBaseUtils/SofaBaseUtils_test/CMakeLists.txt;19;add_test;C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaBaseUtils/SofaBaseUtils_test/CMakeLists.txt;0;")
else()
  add_test(SofaBaseUtils_test NOT_AVAILABLE)
endif()
