cmake_minimum_required(VERSION 3.10)

project(text2vec)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

include_directories(
    ./
    /usr/include/zdl
    ${CMAKE_SOURCE_DIR}/3rd/spdlog/include
    ${JSONCPP_INCLUDE_DIRS}
)

link_directories(
    ${JSONCPP_LIBRARY_DIRS}
)

# Config Logger
if(NOT DEFINED LOG_LEVEL)
    message(STATUS "Not define log print level, default is 'info'")
    set(LOG_LEVEL "info")
endif()
add_definitions(-DLOG_LEVEL="${LOG_LEVEL}")
message(STATUS "log level: ${LOG_LEVEL}")

option(DUMP_LOG "Dump log into a file." OFF)
option(MULTI_LOG "Dump log and stdout." OFF)

if(DUMP_LOG OR MULTI_LOG)
    if(NOT DEFINED LOG_PATH)
        message(STATUS "Not define log path, use default")
        set(LOG_PATH "./log")
        message(STATUS "log path: ${LOG_PATH}")
    endif()
    if(NOT DEFINED LOG_FILE_PREFIX)
        message(STATUS "Not define log name prefix, use default")
        set(LOG_FILE_PREFIX ${PROJECT_NAME})
        message(STATUS "log file prefix: ${LOG_FILE_PREFIX}")
    endif()

    add_definitions(
        -DDUMP_LOG
        -DLOG_PATH="${LOG_PATH}"
        -DLOG_FILE_PREFIX="${LOG_FILE_PREFIX}"
    )
    if(MULTI_LOG)
        message(STATUS "Multi log set.")
        add_definitions(-DMULTI_LOG)
    endif()
endif()

add_library(mobilebert
    SHARED
    tokenization.cpp
    Bert.cpp
    SNPETask.cpp
)

target_link_libraries(mobilebert
    pthread
    dl
    utf8proc
    SNPE
)

add_executable(test-text2vec
    main.cpp
)

target_link_libraries(test-text2vec
    PUBLIC
    pthread
    dl
    utf8proc
    jsoncpp
    SNPE
    mobilebert
)