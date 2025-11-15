# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(ex00_hello_dvr_course LANGUAGES C CXX)

include(../common/cmake/common.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp deviceCode.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE dvr_course-common)
configure_cuda_src(deviceCode.cu)
