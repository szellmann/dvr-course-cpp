# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(ex05_tets_n_friends LANGUAGES C CXX)

include(../common/cmake/common.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME})

if (DVR_COURSE_WITH_CUDA)
  enable_language(CUDA)
else()
  configure_cuda_src(deviceCode.cu)
endif()

if (DVR_COURSE_WITH_FAKE_OWL)
  fake_owl_compile_and_embed(ptxCode
    deviceCode.cu
    LINK_LIBRARIES dvr_course-common
  )
  target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp ${ptxCode})
elseif (DVR_COURSE_WITH_OWL)
  target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp)
  embed_ptx(
    OUTPUT_TARGET ex05_deviceCode-ptx
    PTX_LINK_LIBRARIES dvr_course-common
    EMBEDDED_SYMBOL_NAMES ptxCode
    SOURCES deviceCode.cu
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ex05_deviceCode-ptx)
else()
  target_sources(${PROJECT_NAME} PRIVATE hostCode.cpp deviceCode.cu)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE dvr_course-common)
