# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(ex02_woodcock LANGUAGES C CXX)

include(../common/cmake/common.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME})
if (DVR_COURSE_WITH_OWL)
  enable_language(CUDA)
  target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp)
  set_source_files_properties(deviceCode.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr")
  embed_ptx(
    OUTPUT_TARGET ex02_deviceCode-ptx
    PTX_LINK_LIBRARIES dvr_course-common $<BUILD_INTERFACE:nanovdb>
    SOURCES deviceCode.cu
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ex02_deviceCode-ptx)
else()
  target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp deviceCode.cu)
  configure_cuda_src(deviceCode.cu)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE dvr_course-common)
target_link_libraries(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:nanovdb>)
