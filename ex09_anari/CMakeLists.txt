# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(ex09_anari LANGUAGES C CXX)

include(../common/cmake/common.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari REQUIRED)

# ANARI app:
add_executable(${PROJECT_NAME}_app)
target_sources(${PROJECT_NAME}_app PRIVATE app/hostCode.cpp)
target_link_libraries(${PROJECT_NAME}_app PRIVATE dvr_course-common anari::anari)

# ANARI "device" (integrates the code we developed in the course into a library
# that can interoperate with external tools):
add_library(anari_library_${PROJECT_NAME} SHARED)
target_sources(anari_library_${PROJECT_NAME} PRIVATE
  device/Device.cpp
  device/Library.cpp
  device/hostCode.cpp
)

anari_generate_queries(
  DEVICE_TARGET anari_library_${PROJECT_NAME}
  CPP_NAMESPACE ex09_anari
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/device/ex09_anari_device.json
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/device
)

include(GenerateExportHeader)
generate_export_header(anari_library_${PROJECT_NAME}
  EXPORT_MACRO_NAME "DVR_COURSE_DEVICE_INTERFACE"
)

if (DVR_COURSE_WITH_CUDA)
  enable_language(CUDA)
else()
  configure_cuda_src(device/deviceCode.cu)
endif()

if (DVR_COURSE_WITH_FAKE_OWL)
  fake_owl_compile_and_embed(ptxCode
    device/deviceCode.cu
    LINK_LIBRARIES dvr_course-common
  )
  target_sources(anari_library_${PROJECT_NAME} PRIVATE ${ptxCode})
elseif (DVR_COURSE_WITH_OWL)
  set_source_files_properties(device/deviceCode.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr")
  embed_ptx(
    OUTPUT_TARGET ex09_deviceCode-ptx
    PTX_LINK_LIBRARIES dvr_course-common
    EMBEDDED_SYMBOL_NAMES ptxCode
    SOURCES device/deviceCode.cu
  )
  target_link_libraries(anari_library_${PROJECT_NAME} PRIVATE ex09_deviceCode-ptx)
else()
  target_sources(anari_library_${PROJECT_NAME} PRIVATE device/deviceCode.cu)
endif()
target_link_libraries(anari_library_${PROJECT_NAME}
PUBLIC 
  dvr_course-common
  anari::anari
  anari::helium
)
target_include_directories(anari_library_${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
