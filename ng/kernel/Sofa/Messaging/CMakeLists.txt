cmake_minimum_required(VERSION 3.5)
project(Sofa.Messaging)
string(TOUPPER ${PROJECT_NAME} PROJECT_UNAME)
string(MAKE_C_IDENTIFIER ${PROJECT_UNAME} PROJECT_CNAME)

find_package(Sofa.Config)

set(HEADER_FILES
    src/sofa/messaging/Messaging.h
    src/sofa/messaging/Message.h
    src/sofa/messaging/ComponentInfo.h
    src/sofa/messaging/FileInfo.h
    src/sofa/messaging/MessageDispatcher.h
    src/sofa/messaging/MessageHandler.h
    src/sofa/messaging/SilentMessageHandler.h
    src/sofa/messaging/MessageHandler.h
    src/sofa/messaging/MessageFormatter.h
)


set(SOURCE_FILES
    src/sofa/messaging/Message.cpp
    src/sofa/messaging/MessageDispatcher.cpp
    src/sofa/messaging/ComponentInfo.cpp
    src/sofa/messaging/SilentMessageHandler.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_${PROJECT_CNAME}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

message("TARGET : " ${PROJECT_CNAME})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
